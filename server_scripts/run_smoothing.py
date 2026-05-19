"""
Manifold smoothing in concept space.

Steps:
1. Load CFM model (CLIP-DINOiser + SAE)
2. Load probe dataset (ImageNet/Places365) + linear classifier
3. Generate concept vectors for all images (cached to disk)
4. Build KNN index on concept vectors (cached to disk)
5. For target images: KNN -> PCA -> whiten -> noise -> classify -> majority vote

Usage:
  python server_scripts/run_smoothing.py

  # or as SLURM job:
  sbatch server_scripts/smoothing_job.sh
"""

import sys
import os
sys.stdout.reconfigure(line_buffering=True)  # force line-buffered output for SLURM
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
import json
from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA
from scipy.stats import norm, binom, pearsonr, kendalltau
from scipy.special import gammaln
import matplotlib
matplotlib.use('Agg')  # no display on server
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from cfm.arg_parser import get_default_parser
from cfm.utils import common_init, get_probe_dataset
from cfm import config as cfm_config
from cfm.method_utils import MethodCFM
from cfm.data_utils import probe_classnames

# ===========================================================================
# Config
# ===========================================================================
CONFIG_NAME = 'k_12_ef_16_lr_0.0001_mf_[0.008,0.03,0.06,0.12,0.24,0.542]'
PROBE_DATASET = "imagenet"       # or "places365"
PROBE_SPLIT = "val"
PROBE_CONFIG = "lr0.0001_bs512_epo50_clCE_spL1_spl0.0max_no_threshold"

# Smoothing parameters
K_NEIGHBORS = 500
SCALE_WEIGHT = float(os.environ.get('CFM_SIGMA', '0.7'))  # supports multi-sigma via env
N_SMOOTH_SAMPLES = 100
N_TARGETS = 500           # number of val images to certify
N_VIZ = 10                # save detailed viz/JSON only for the first N images

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'smoothing_data')  # shared artifacts (vectors, index)
SAVE_DIR = os.path.join(DATA_DIR, f'sigma_{SCALE_WEIGHT:.2f}')  # sigma-specific results
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# Helpers
# ===========================================================================

# --- Cohen et al. certification helpers ---
CERTIFY_ALPHA = 0.001  # confidence level for Clopper-Pearson bounds

def clopper_pearson_lower(n_success, n_total, alpha=CERTIFY_ALPHA):
    """Lower bound on success probability (Clopper-Pearson)."""
    if n_success == 0:
        return 0.0
    return binom.ppf(alpha / 2, n_success, 1.0) / n_total if n_success > 0 else 0.0

def clopper_pearson_upper(n_success, n_total, alpha=CERTIFY_ALPHA):
    """Upper bound on success probability (Clopper-Pearson)."""
    if n_success == n_total:
        return 1.0
    return binom.ppf(1 - alpha / 2, n_success + 1, 1.0) / n_total

def certified_radius(sigma, p_a_lower, p_b_upper):
    """Cohen et al. certified L2 radius: r = σ/2 * (Φ⁻¹(p_A_lower) - Φ⁻¹(p_B_upper))"""
    if p_a_lower <= p_b_upper or p_a_lower <= 0.5:
        return 0.0  # abstain
    return sigma / 2.0 * (norm.ppf(p_a_lower) - norm.ppf(p_b_upper))

def certify_class_votes(vote_counts, n_total, sigma):
    """Certify a single class prediction from vote counts.
    Returns dict with p_A_lower, p_B_upper, abstained, certified_radius, top_class, n_votes.
    """
    sorted_votes = vote_counts.most_common()
    top_class, n_a = sorted_votes[0]
    n_b = sorted_votes[1][1] if len(sorted_votes) > 1 else 0

    p_a_lower = clopper_pearson_lower(n_a, n_total)
    p_b_upper = clopper_pearson_upper(n_b, n_total)
    abstained = bool(p_a_lower <= p_b_upper)
    radius = certified_radius(sigma, p_a_lower, p_b_upper)

    return {
        'top_class': top_class,
        'n_votes': n_a,
        'n_votes_runner_up': n_b,
        'p_a_lower': round(p_a_lower, 6),
        'p_b_upper': round(p_b_upper, 6),
        'abstained': abstained,
        'certified_radius': round(radius, 6),
    }

def certify_concept(n_survived, n_total, sigma):
    """Certify a single concept's presence in top-K.
    Treats concept-in-top-K as a binary classification: 'present' vs 'absent'.
    Returns dict with survival_rate, p_a_lower, certified, radius.
    """
    n_absent = n_total - n_survived
    p_a_lower = clopper_pearson_lower(n_survived, n_total)
    p_b_upper = clopper_pearson_upper(n_absent, n_total)
    is_certified = bool((p_a_lower > 0.5) and (not (p_a_lower <= p_b_upper)))
    radius = certified_radius(sigma, p_a_lower, p_b_upper)

    return {
        'survival_rate': round(n_survived / n_total, 4),
        'p_a_lower': round(p_a_lower, 6),
        'certified': is_certified,
        'certified_radius': round(radius, 6),
    }


def compute_concept_scores(cv_orig, cv_smoothed_avg, top_k=12):
    """Paper-ready concept stability metrics.

    Returns dict with:
      concept_fidelity   – Pearson r between original and smoothed vectors
      rank_correlation   – Kendall τ over the top-k original concept activations
      concept_drift      – relative L2:  ||Δ|| / ||orig||
      spurious_act_rate  – fraction of smoothed top-k that were NOT in original top-k
      act_drop_score     – mean activation drop for original top-k concepts
      act_gain_score     – mean activation gain for originally-zero concepts that became active
    """
    o = np.asarray(cv_orig, dtype=np.float64)
    s = np.asarray(cv_smoothed_avg, dtype=np.float64)

    # --- Concept Fidelity (Pearson r) ---
    if np.std(o) < 1e-12 or np.std(s) < 1e-12:
        fidelity = 0.0
    else:
        fidelity, _ = pearsonr(o, s)

    # --- Rank Correlation (Kendall τ over original top-k) ---
    orig_topk = np.argsort(-o)[:top_k]
    if len(orig_topk) > 1:
        tau, _ = kendalltau(o[orig_topk], s[orig_topk])
        tau = 0.0 if np.isnan(tau) else tau
    else:
        tau = 0.0

    # --- Concept Drift (relative L2) ---
    norm_o = np.linalg.norm(o)
    drift = float(np.linalg.norm(o - s) / norm_o) if norm_o > 1e-12 else 0.0

    # --- Spurious Activation Rate ---
    smooth_topk = set(np.argsort(-s)[:top_k].tolist())
    orig_topk_set = set(orig_topk.tolist())
    sar = len(smooth_topk - orig_topk_set) / top_k

    # --- Activation Drop Score (mean drop for original top-k) ---
    drops = np.clip(o[orig_topk] - s[orig_topk], 0, None)
    ads = float(drops.mean())

    # --- Activation Gain Score (mean gain for originally-zero → active) ---
    zero_mask = o < 1e-8
    gained = s[zero_mask]
    ags = float(gained[gained > 1e-8].mean()) if (gained > 1e-8).any() else 0.0

    return {
        'concept_fidelity': round(fidelity, 6),
        'rank_correlation': round(tau, 6),
        'concept_drift': round(drift, 6),
        'spurious_act_rate': round(sar, 4),
        'act_drop_score': round(ads, 6),
        'act_gain_score': round(ags, 6),
    }


def log_unit_ball_volume(d: int) -> float:
    """Log of volume of unit ball in d dimensions: log(π^(d/2) / Γ(d/2+1))."""
    return (d / 2.0) * np.log(np.pi) - gammaln(d / 2.0 + 1)


def compute_volumes(r_iso: float, r_mani: float, eigenvalues: np.ndarray, D: int) -> dict:
    """Compute 4 certified volume quantities (log-space) per Jonas's framework.

    Qty 1: Ambient Iso Ball         = C_D · r_iso^D
    Qty 2: Projected Iso Ball       = C_k · r_iso^k
    Qty 3: Manifold-aware (iso r)   = C_k · r_iso^k · √det(Λ)
    Qty 4: Manifold Ellipsoid       = C_k · r_mani^k · √det(Λ)
    """
    k = len(eigenvalues)
    log_C_D = log_unit_ball_volume(D)
    log_C_k = log_unit_ball_volume(k)
    log_det_half = 0.5 * np.sum(np.log(eigenvalues + 1e-30))  # √det(Λ) in log

    log_qty1 = (log_C_D + D * np.log(r_iso)) if r_iso > 0 else -np.inf
    log_qty2 = (log_C_k + k * np.log(r_iso)) if r_iso > 0 else -np.inf
    log_qty3 = (log_C_k + k * np.log(r_iso) + log_det_half) if r_iso > 0 else -np.inf
    log_qty4 = (log_C_k + k * np.log(r_mani) + log_det_half) if r_mani > 0 else -np.inf

    return {
        'log_vol_iso_D': round(float(log_qty1), 4),     # Qty 1
        'log_vol_iso_k': round(float(log_qty2), 4),     # Qty 2
        'log_vol_mani_pred': round(float(log_qty3), 4), # Qty 3
        'log_vol_mani_actual': round(float(log_qty4), 4), # Qty 4
        'r_iso': round(float(r_iso), 6),
        'r_mani': round(float(r_mani), 6),
        'k': k,
        'D': D,
    }


def classify_concept_vector(cv, classifier_weights):
    logits = cv @ classifier_weights.T
    return logits.argmax().item()


def get_class_name(probe_dataset, class_idx):
    names = probe_classnames.probe_classes_dict[probe_dataset]
    if probe_dataset == "places365":
        return " ".join(names[class_idx].split("/")[2:]).replace("_", " ")
    return names[class_idx]


def get_top_concept_info(cv, concept_names, top_k=10):
    vals, idxs = torch.topk(torch.tensor(cv), top_k)
    results = []
    for i, v in zip(idxs, vals):
        name = concept_names[i.item()] if concept_names else f"concept_{i.item()}"
        results.append((name, v.item()))
    return results


def get_concept_changes(cv_orig, cv_smoothed, concept_names, top_k=10):
    """Compute top drops, gains, and least activated (originally active) concepts."""
    diff = np.array(cv_orig) - np.array(cv_smoothed)
    cv_o = np.array(cv_orig)
    cv_s = np.array(cv_smoothed)
    # Top drops: concepts that decreased the most
    drop_idxs = np.argsort(-diff)[:top_k]  # largest positive diff = biggest drop
    drops = [(concept_names[i] if concept_names else f"c_{i}",
              round(float(diff[i]), 4), round(float(cv_o[i]), 4), round(float(cv_s[i]), 4))
             for i in drop_idxs if diff[i] > 0]
    # Top gains: concepts that increased the most in magnitude after smoothing
    gain_idxs = np.argsort(diff)[:top_k]  # largest negative diff = biggest gain
    gains = [(concept_names[i] if concept_names else f"c_{i}",
              round(float(-diff[i]), 4), round(float(cv_o[i]), 4), round(float(cv_s[i]), 4))
             for i in gain_idxs if diff[i] < 0]
    # Least activated: originally active concepts with lowest smoothed activation
    active_mask = cv_o > 1e-6
    if active_mask.sum() > 0:
        active_idxs = np.where(active_mask)[0]
        active_smoothed = cv_s[active_idxs]
        order = np.argsort(active_smoothed)[:top_k]
        least = [(concept_names[active_idxs[j]] if concept_names else f"c_{active_idxs[j]}",
                  round(float(cv_s[active_idxs[j]]), 4), round(float(cv_o[active_idxs[j]]), 4))
                 for j in order]
    else:
        least = []
    return drops, gains, least


# ===========================================================================
# Step 1: Load config + concept names + classifier (NO CFM model needed)
# ===========================================================================
print("=" * 70)
print("Loading config, concept names, and classifier")
print("=" * 70)

parser = get_default_parser()
args = parser.parse_args([])
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
common_init(args)
args.config_name = CONFIG_NAME

print(f"Device: {args.device}")

# Concept names
from cfm import config as cfg
concept_name_save_path = getattr(cfg, 'concept_names_override', None)
if not concept_name_save_path or not os.path.exists(concept_name_save_path):
    concept_name_save_path = os.path.join(
        str(args.save_dir_sae_ckpts['img']),
        args.save_suffix,
        args.config_name,
        'trainer_0',
        'concept_names.txt'
    )
if os.path.exists(concept_name_save_path):
    with open(concept_name_save_path, "r") as f:
        concept_names = [line.strip() for line in f.readlines()]
    print(f"Loaded {len(concept_names)} concept names")
else:
    concept_names = None
    print("Concept names not found, using indices")

# Linear probe classifier
args.probe_dataset = PROBE_DATASET
args.probe_split = PROBE_SPLIT
args.probe_dataset_root_dir = cfm_config.probe_dataset_root_dir_dict[PROBE_DATASET]

if not hasattr(args, "autoencoder_input_dim_dict"):
    args.autoencoder_input_dim_dict = cfm_config.autoencoder_input_dim_dict

method_obj = MethodCFM(args, vocab_txt_path=None)

probe_base = os.path.join(
    args.probe_cs_save_dir_root, args.sae_dataset,
    args.img_enc_name_for_saving, args.hook_points[0],
    args.config_name, PROBE_DATASET, PROBE_CONFIG, "on_concepts_ckpts")
ckpt_file = [f for f in os.listdir(probe_base) if f.endswith(".pt")][0]
ckpt_path = os.path.join(probe_base, ckpt_file)
print(f"Loading probe from: {ckpt_path}")

classifier_weights = method_obj.get_classifier_weights(
    probe_dataset=PROBE_DATASET, checkpoint_save_path=ckpt_path).to(args.device)
print(f"Classifier weights: {classifier_weights.shape}")


# ===========================================================================
# Step 2: Load cached concept vectors (built by build_knn_index.py)
# ===========================================================================
print("\n" + "=" * 70)
print("Loading cached concept vectors")
print("=" * 70)

train_cv_path = os.path.join(DATA_DIR, f"concept_vectors_{PROBE_DATASET}_train.pt")
train_labels_path = os.path.join(DATA_DIR, f"labels_{PROBE_DATASET}_train.pt")
val_cv_path = os.path.join(DATA_DIR, f"concept_vectors_{PROBE_DATASET}_val.pt")
val_labels_path = os.path.join(DATA_DIR, f"labels_{PROBE_DATASET}_val.pt")

for p in [train_cv_path, train_labels_path, val_cv_path, val_labels_path]:
    if not os.path.exists(p):
        print(f"ERROR: Missing {p}")
        print("Run build_knn_index.py first to generate concept vectors and KNN index.")
        sys.exit(1)

train_concept_vectors = torch.load(train_cv_path)
train_labels = torch.load(train_labels_path)
print(f"Train: {train_concept_vectors.shape[0]} vectors, dim={train_concept_vectors.shape[1]}")

val_concept_vectors = torch.load(val_cv_path)
val_labels = torch.load(val_labels_path)
print(f"Val: {val_concept_vectors.shape[0]} vectors")

# For image path lookup we still need the dataset object
preprocess = None  # not needed for smoothing, but needed for dataset path lookup
try:
    from torchvision import transforms
    preprocess = transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
    probe_val_dataset = get_probe_dataset(
        PROBE_DATASET, PROBE_SPLIT, args.probe_dataset_root_dir, preprocess_fn=preprocess)
except Exception:
    probe_val_dataset = None
    print("Warning: could not load val dataset for image path lookup")


# ===========================================================================
# Step 3: Load pre-built KNN index (built by build_knn_index.py)
# ===========================================================================
print("\n" + "=" * 70)
print("Loading KNN index")
print("=" * 70)

import annoy
import glob

N_TRAIN = train_concept_vectors.shape[0]
CONCEPT_DIM = train_concept_vectors.shape[1]

index_pattern = os.path.join(DATA_DIR, f"knn_concepts_{PROBE_DATASET}_train_*.ann")
available_indices = sorted(glob.glob(index_pattern), key=os.path.getsize, reverse=True)

index_path = None
for candidate in available_indices:
    try:
        n_in_index = int(os.path.basename(candidate).split("_train_")[1].replace(".ann", ""))
        if n_in_index >= N_TRAIN:
            index_path = candidate
            break
    except (ValueError, IndexError):
        continue

# Fall back to any available index
if index_path is None and available_indices:
    index_path = available_indices[0]
    print(f"WARNING: No full-size index found, using best available: {index_path}", flush=True)

if index_path is None or not os.path.exists(index_path):
    print("ERROR: No KNN index found!")
    print("Run build_knn_index.py first to build the index.")
    sys.exit(1)

knn_index = annoy.AnnoyIndex(CONCEPT_DIM, 'euclidean')
knn_index.load(index_path)
print(f"Loaded KNN index from {index_path}", flush=True)


# ===========================================================================
# Step 5: Manifold smoothing
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("Manifold smoothing", flush=True)
print("=" * 70, flush=True)
print(f"K={K_NEIGHBORS}, sigma={SCALE_WEIGHT}, N_samples={N_SMOOTH_SAMPLES}", flush=True)
print(f"Index: TRAIN ({N_TRAIN} vectors), Targets: VAL", flush=True)

# --- Incremental saving: write each result as a JSONL line so nothing is lost on OOM ---
results_jsonl_path = os.path.join(SAVE_DIR, f"smoothing_results_{PROBE_DATASET}.jsonl")

# Resume support: load already-completed target indices
completed_idxs = set()
if os.path.exists(results_jsonl_path):
    with open(results_jsonl_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    completed_idxs.add(r['idx'])
                except json.JSONDecodeError:
                    pass
    print(f"Resuming: {len(completed_idxs)} targets already completed", flush=True)

# Open JSONL in append mode — each result is flushed immediately
results_jsonl_file = open(results_jsonl_path, 'a')

# Try to get image paths from the dataset
def get_image_path(dataset, idx):
    """Try to extract image file path from dataset."""
    try:
        if hasattr(dataset, 'samples'):
            return dataset.samples[idx][0]
        elif hasattr(dataset, 'imgs'):
            return dataset.imgs[idx][0]
        elif hasattr(dataset, 'dataset') and hasattr(dataset.dataset, 'samples'):
            return dataset.dataset.samples[idx][0]
    except Exception:
        pass
    return None

# Save dirs per method
manifold_dir = os.path.join(SAVE_DIR, f"manifold_{PROBE_DATASET}")
isotropic_dir = os.path.join(SAVE_DIR, f"isotropic_{PROBE_DATASET}")
os.makedirs(manifold_dir, exist_ok=True)
os.makedirs(isotropic_dir, exist_ok=True)

np.random.seed(42)
TARGET_IDCS = np.random.choice(len(val_concept_vectors), size=N_TARGETS, replace=False).tolist()
print(f"Certifying {N_TARGETS} val images (saving viz for first {N_VIZ})", flush=True)
viz_count = 0

for loop_i, target_idx in enumerate(TARGET_IDCS):
    # Skip already-completed targets (resume after OOM/restart)
    if target_idx in completed_idxs:
        if (loop_i + 1) % 50 == 0:
            print(f"  [{loop_i+1}/{N_TARGETS}] skipped (already done)", flush=True)
        continue

    cv_orig = val_concept_vectors[target_idx].numpy()
    label_true = val_labels[target_idx].item()
    pred_orig = classify_concept_vector(
        torch.tensor(cv_orig, device=args.device), classifier_weights)

    # Save input image path (and copy image if possible)
    img_path = get_image_path(probe_val_dataset, target_idx)

    # KNN neighbors from TRAIN index (query by vector)
    nn_idcs = knn_index.get_nns_by_vector(cv_orig.tolist(), K_NEIGHBORS)
    X_neighbors = np.stack([train_concept_vectors[i].numpy() for i in nn_idcs])

    # --- Top 5 neighbors: save their concepts ---
    top5_neighbors_info = []
    for rank, nn_idx in enumerate(nn_idcs[:5]):
        nn_cv = train_concept_vectors[nn_idx].numpy()
        nn_label = train_labels[nn_idx].item()
        nn_pred = classify_concept_vector(
            torch.tensor(nn_cv, device=args.device), classifier_weights)
        nn_concepts = get_top_concept_info(nn_cv, concept_names, top_k=20)
        top5_neighbors_info.append({
            'rank': rank + 1,
            'dataset_idx': int(nn_idx),
            'true_class': get_class_name(PROBE_DATASET, nn_label),
            'pred_class': get_class_name(PROBE_DATASET, nn_pred),
            'top_concepts': [{'name': n, 'value': round(v, 4)} for n, v in nn_concepts],
        })
    # PCA on neighborhood
    mean_nn = X_neighbors.mean(axis=0)
    X_centered = X_neighbors - mean_nn
    pca = PCA(n_components=K_NEIGHBORS)
    pca.fit(X_centered)
    ev = pca.explained_variance_
    Vt = pca.components_

    # Whiten original
    cv_whitened = (cv_orig) @ Vt.T / np.sqrt(ev)

    # Smoothing loop — collect per-concept survival + downstream class votes
    smooth_preds = []
    smooth_concepts_overlap = []
    example_noisy_samples = []
    orig_top_k = 20
    orig_top_idxs = np.argsort(-cv_orig)[:orig_top_k]  # indices of top-K active concepts
    orig_top = set(orig_top_idxs.tolist())
    # Per-concept survival counter: how many of N samples still have this concept in top-K
    concept_survival_counts = {int(idx): 0 for idx in orig_top_idxs}
    # Track ALL concepts that appear in top-K across any smooth sample (for vote histogram)
    all_concept_votes_manifold = Counter()
    # Track activation values of original top-K concepts across all smooth samples (for distribution plot)
    manifold_activation_traces = {int(idx): [] for idx in orig_top_idxs}
    # Also store ALL smoothed vectors for viz (to analyze least-activated/new concepts)
    save_viz = (viz_count < N_VIZ)
    all_manifold_smooth_vecs = [] if save_viz else None

    for sample_i in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, SCALE_WEIGHT, size=len(ev))
        cv_noised = cv_whitened + noise
        cv_smoothed = cv_noised @ (np.sqrt(ev)[:, None] * Vt) + mean_nn

        # Record activation of each original top-K concept in this smooth sample
        for cidx in orig_top_idxs:
            manifold_activation_traces[int(cidx)].append(float(cv_smoothed[cidx]))
        if all_manifold_smooth_vecs is not None:
            all_manifold_smooth_vecs.append(cv_smoothed)

        # Downstream class prediction
        pred_i = classify_concept_vector(
            torch.tensor(cv_smoothed, dtype=torch.float32, device=args.device),
            classifier_weights)
        smooth_preds.append(pred_i)

        # Concept-level: which of the original top-K survived?
        smoothed_top_idxs = np.argsort(-cv_smoothed)[:orig_top_k]
        smoothed_top = set(smoothed_top_idxs.tolist())
        overlap = len(orig_top & smoothed_top) / orig_top_k
        smooth_concepts_overlap.append(overlap)
        for cidx in orig_top_idxs:
            if int(cidx) in smoothed_top:
                concept_survival_counts[int(cidx)] += 1
        # Count ALL concepts in top-K (including new ones)
        for cidx in smoothed_top_idxs:
            all_concept_votes_manifold[int(cidx)] += 1

        # Save first 5 noisy samples as examples
        if sample_i < 5:
            sample_concepts = get_top_concept_info(cv_smoothed, concept_names, top_k=orig_top_k)
            example_noisy_samples.append({
                'sample_idx': sample_i + 1,
                'pred_class': get_class_name(PROBE_DATASET, pred_i),
                'top20_overlap': round(overlap, 3),
                'top_concepts': [{'name': n, 'value': round(v, 4)} for n, v in sample_concepts],
            })

    # Per-concept certification (Clopper-Pearson + radius)
    concept_survival_named = []
    for cidx in orig_top_idxs:
        cname = concept_names[cidx] if concept_names else str(cidx)
        n_survived = concept_survival_counts[int(cidx)]
        cert = certify_concept(n_survived, N_SMOOTH_SAMPLES, SCALE_WEIGHT)
        concept_survival_named.append({
            'concept_idx': int(cidx),
            'name': cname,
            'orig_activation': round(float(cv_orig[cidx]), 4),
            'n_survived': n_survived,
            **cert,
        })
    n_concepts_certified = sum(1 for c in concept_survival_named if c['certified'])
    mean_concept_survival = round(float(np.mean([c['survival_rate'] for c in concept_survival_named])), 4)
    concept_radii = [c['certified_radius'] for c in concept_survival_named if c['certified']]
    min_concept_radius = round(min(concept_radii), 6) if concept_radii else 0.0
    mean_concept_radius = round(float(np.mean(concept_radii)), 6) if concept_radii else 0.0

    # Downstream class certification (Cohen et al.)
    vote_counts = Counter(smooth_preds)
    class_cert = certify_class_votes(vote_counts, N_SMOOTH_SAMPLES, SCALE_WEIGHT)
    pred_smooth = class_cert['top_class']
    n_votes = class_cert['n_votes']

    # --- Compute "average smoothed" concept vector ---
    # Re-run to get the mean smoothed vector for final concept summary
    cv_smoothed_accum = np.zeros_like(cv_orig)
    for _ in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, SCALE_WEIGHT, size=len(ev))
        cv_noised = cv_whitened + noise
        cv_smoothed_accum += cv_noised @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
    cv_smoothed_avg = cv_smoothed_accum / N_SMOOTH_SAMPLES
    final_concepts = get_top_concept_info(cv_smoothed_avg, concept_names, top_k=20)

    # --- Baseline: Isotropic Gaussian smoothing (no manifold) ---
    gauss_sigma = SCALE_WEIGHT  # same sigma as manifold for fair comparison
    gauss_preds = []
    gauss_overlap = []
    gauss_example_samples = []
    gauss_concept_survival_counts = {int(idx): 0 for idx in orig_top_idxs}
    all_concept_votes_gaussian = Counter()
    gauss_activation_traces = {int(idx): [] for idx in orig_top_idxs}
    all_gauss_smooth_vecs = [] if save_viz else None

    for sample_i in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
        cv_gauss = cv_orig + noise

        for cidx in orig_top_idxs:
            gauss_activation_traces[int(cidx)].append(float(cv_gauss[cidx]))
        if all_gauss_smooth_vecs is not None:
            all_gauss_smooth_vecs.append(cv_gauss)
        pred_g = classify_concept_vector(
            torch.tensor(cv_gauss, dtype=torch.float32, device=args.device),
            classifier_weights)
        gauss_preds.append(pred_g)
        gauss_top_idxs_arr = np.argsort(-cv_gauss)[:orig_top_k]
        gauss_top = set(gauss_top_idxs_arr.tolist())
        g_overlap = len(orig_top & gauss_top) / orig_top_k
        gauss_overlap.append(g_overlap)
        for cidx in orig_top_idxs:
            if int(cidx) in gauss_top:
                gauss_concept_survival_counts[int(cidx)] += 1
        for cidx in gauss_top_idxs_arr:
            all_concept_votes_gaussian[int(cidx)] += 1
        if sample_i < 5:
            g_concepts = get_top_concept_info(cv_gauss, concept_names, top_k=orig_top_k)
            gauss_example_samples.append({
                'sample_idx': sample_i + 1,
                'pred_class': get_class_name(PROBE_DATASET, pred_g),
                'top20_overlap': round(g_overlap, 3),
                'top_concepts': [{'name': n, 'value': round(v, 4)} for n, v in g_concepts],
            })

    gauss_concept_survival_named = []
    for cidx in orig_top_idxs:
        cname = concept_names[cidx] if concept_names else str(cidx)
        n_survived = gauss_concept_survival_counts[int(cidx)]
        cert = certify_concept(n_survived, N_SMOOTH_SAMPLES, gauss_sigma)
        gauss_concept_survival_named.append({
            'concept_idx': int(cidx),
            'name': cname,
            'orig_activation': round(float(cv_orig[cidx]), 4),
            'n_survived': n_survived,
            **cert,
        })
    g_n_concepts_certified = sum(1 for c in gauss_concept_survival_named if c['certified'])
    g_mean_concept_survival = round(float(np.mean([c['survival_rate'] for c in gauss_concept_survival_named])), 4)
    g_concept_radii = [c['certified_radius'] for c in gauss_concept_survival_named if c['certified']]
    g_min_concept_radius = round(min(g_concept_radii), 6) if g_concept_radii else 0.0
    g_mean_concept_radius = round(float(np.mean(g_concept_radii)), 6) if g_concept_radii else 0.0

    gauss_vote_counts = Counter(gauss_preds)
    gauss_class_cert = certify_class_votes(gauss_vote_counts, N_SMOOTH_SAMPLES, gauss_sigma)
    pred_gauss = gauss_class_cert['top_class']
    n_votes_gauss = gauss_class_cert['n_votes']

    cv_gauss_accum = np.zeros_like(cv_orig)
    for _ in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
        cv_gauss_accum += cv_orig + noise
    cv_gauss_avg = cv_gauss_accum / N_SMOOTH_SAMPLES
    gauss_final_concepts = get_top_concept_info(cv_gauss_avg, concept_names, top_k=20)

    # --- Print ---
    orig_concepts = get_top_concept_info(cv_orig, concept_names, top_k=20)

    print(f"\n[{loop_i+1}/{N_TARGETS}] idx={target_idx}  "
          f"true={get_class_name(PROBE_DATASET, label_true)}  "
          f"orig={get_class_name(PROBE_DATASET, pred_orig)}  "
          f"manifold={get_class_name(PROBE_DATASET, pred_smooth)}({n_votes}/{N_SMOOTH_SAMPLES})  "
          f"gauss={get_class_name(PROBE_DATASET, pred_gauss)}({n_votes_gauss}/{N_SMOOTH_SAMPLES})  "
          f"\n    class: m_r={class_cert['certified_radius']:.4f}{'(ABSTAIN)' if class_cert['abstained'] else ''}  "
          f"g_r={gauss_class_cert['certified_radius']:.4f}{'(ABSTAIN)' if gauss_class_cert['abstained'] else ''}"
          f"\n    concepts: m_surv={mean_concept_survival:.2f} ({n_concepts_certified}/{orig_top_k} cert, "
          f"min_r={min_concept_radius:.4f})  "
          f"g_surv={g_mean_concept_survival:.2f} ({g_n_concepts_certified}/{orig_top_k} cert, "
          f"min_r={g_min_concept_radius:.4f})", flush=True)

    if save_viz:
        print(f"\n  ORIGINAL top-20 concepts:")
        for name, val in orig_concepts:
            print(f"    {name:30s} {val:.4f}")

        print(f"\n  TOP 5 NEIGHBORS:")
        for nb in top5_neighbors_info:
            print(f"    #{nb['rank']} idx={nb['dataset_idx']}  "
                  f"class={nb['true_class']}  pred={nb['pred_class']}")
            for c in nb['top_concepts'][:20]:
                print(f"        {c['name']:30s} {c['value']:.4f}")

        print(f"\n  5 EXAMPLE NOISY SAMPLES:")
        for s in example_noisy_samples:
            print(f"    Sample {s['sample_idx']}  pred={s['pred_class']}  "
                  f"overlap={s['top20_overlap']}")
            for c in s['top_concepts'][:20]:
                print(f"        {c['name']:30s} {c['value']:.4f}")

        print(f"\n  FINAL AVERAGED CONCEPTS (after {N_SMOOTH_SAMPLES}x smoothing):")
        for name, val in final_concepts:
            print(f"    {name:30s} {val:.4f}")

        print(f"\n  --- BASELINE: Isotropic Gaussian (σ={gauss_sigma:.4f}) ---")
        print(f"  5 EXAMPLE GAUSSIAN SAMPLES:")
        for s in gauss_example_samples:
            print(f"    Sample {s['sample_idx']}  pred={s['pred_class']}  "
                  f"overlap={s['top20_overlap']}")
            for c in s['top_concepts'][:5]:
                print(f"        {c['name']:30s} {c['value']:.4f}")
        print(f"  GAUSSIAN AVERAGED top-20 concepts:")
        for name, val in gauss_final_concepts:
            print(f"    {name:30s} {val:.4f}")

    # ===================================================================
    # Save detailed viz + JSON only for first N_VIZ images
    # ===================================================================
    if save_viz:
        gauss_top_concepts_pre = get_top_concept_info(cv_gauss_avg, concept_names, top_k=20)

        # --- Compute concept changes for both methods ---
        m_drops, m_gains, m_least = get_concept_changes(cv_orig, cv_smoothed_avg, concept_names, top_k=10)
        g_drops, g_gains, g_least = get_concept_changes(cv_orig, cv_gauss_avg, concept_names, top_k=10)

        # --- Manifold visualization (2x3 grid) ---
        fig, axes = plt.subplots(2, 3, figsize=(26, 16))

        pca_vis = PCA(n_components=2)
        X_vis = pca_vis.fit_transform(X_neighbors)
        orig_vis = pca_vis.transform(cv_orig.reshape(1, -1))[0]
        mean_vis = pca_vis.transform(mean_nn.reshape(1, -1))[0]

        vis_noisy_points = []
        vis_noisy_preds = []
        for _ in range(N_SMOOTH_SAMPLES):
            noise = np.random.normal(0, SCALE_WEIGHT, size=len(ev))
            cv_noised = cv_whitened + noise
            cv_s = cv_noised @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
            vis_noisy_points.append(cv_s)
            p = classify_concept_vector(
                torch.tensor(cv_s, dtype=torch.float32, device=args.device),
                classifier_weights)
            vis_noisy_preds.append(p)
        noisy_vis = pca_vis.transform(np.stack(vis_noisy_points))
        noisy_correct = [p == label_true for p in vis_noisy_preds]

        # Row 1, Panel 1: Manifold neighborhood
        ax = axes[0, 0]
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightblue', s=8, alpha=0.4, label=f'{K_NEIGHBORS} KNN')
        for rank in range(min(5, len(nn_idcs))):
            nn_cv_vis = pca_vis.transform(train_concept_vectors[nn_idcs[rank]].numpy().reshape(1, -1))[0]
            ax.scatter(nn_cv_vis[0], nn_cv_vis[1], c='blue', s=80, marker='D', zorder=5,
                       edgecolors='darkblue', linewidths=1.5)
            ax.annotate(f'N{rank+1}', (nn_cv_vis[0], nn_cv_vis[1]), fontsize=8, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')
        ax.scatter(orig_vis[0], orig_vis[1], c='red', s=200, marker='*', zorder=10,
                   edgecolors='darkred', linewidths=1.5, label='Target')
        ax.scatter(mean_vis[0], mean_vis[1], c='green', s=100, marker='X', zorder=10,
                   edgecolors='darkgreen', linewidths=1.5, label='Mean')
        ax.set_title(f'Manifold Neighborhood (PCA)\nTrue: {get_class_name(PROBE_DATASET, label_true)}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 1, Panel 2: Manifold noisy samples
        ax = axes[0, 1]
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightgray', s=5, alpha=0.2)
        colors = ['green' if c else 'orange' for c in noisy_correct]
        ax.scatter(noisy_vis[:, 0], noisy_vis[:, 1], c=colors, s=15, alpha=0.6,
                   label=f'{N_SMOOTH_SAMPLES} samples')
        ax.scatter(orig_vis[0], orig_vis[1], c='red', s=200, marker='*', zorder=10,
                   edgecolors='darkred', linewidths=1.5, label='Original')
        cov = np.cov(noisy_vis.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
        for n_std in [1, 2]:
            ell = Ellipse(xy=noisy_vis.mean(axis=0), width=2*n_std*np.sqrt(eigenvalues[1]),
                          height=2*n_std*np.sqrt(eigenvalues[0]), angle=angle,
                          fill=False, edgecolor='purple', linestyle='--', linewidth=1.5, alpha=0.6)
            ax.add_patch(ell)
        n_correct = sum(noisy_correct)
        ax.set_title(f'Manifold Samples (σ={SCALE_WEIGHT})\nCorrect: {n_correct}/{N_SMOOTH_SAMPLES}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 1, Panel 3: Manifold concept bar chart (top activations)
        ax = axes[0, 2]
        n_show = 20
        orig_top_concepts = get_top_concept_info(cv_orig, concept_names, top_k=n_show)
        final_top_concepts = get_top_concept_info(cv_smoothed_avg, concept_names, top_k=n_show)
        all_names = []
        for n, _ in orig_top_concepts:
            if n not in all_names: all_names.append(n)
        for n, _ in final_top_concepts:
            if n not in all_names: all_names.append(n)
        all_names = all_names[:25]
        # Build full lookup from vectors (not just top-20) so no concept shows 0 incorrectly
        name_to_idx = {(concept_names[i] if concept_names else f"c_{i}"): i for i in range(len(cv_orig))}
        orig_vals_bar = [float(cv_orig[name_to_idx[n]]) if n in name_to_idx else 0 for n in all_names]
        smooth_vals_bar = [float(cv_smoothed_avg[name_to_idx[n]]) if n in name_to_idx else 0 for n in all_names]
        y_pos = np.arange(len(all_names))
        ax.barh(y_pos - 0.2, orig_vals_bar, height=0.35, color='steelblue', label='Original', alpha=0.8)
        ax.barh(y_pos + 0.2, smooth_vals_bar, height=0.35, color='coral', label='Manifold avg', alpha=0.8)
        ax.set_yticks(y_pos); ax.set_yticklabels([n[:22] for n in all_names], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('Activation')
        ax.set_title(f'Top Activations: {get_class_name(PROBE_DATASET, pred_orig)} → '
                     f'{get_class_name(PROBE_DATASET, pred_smooth)} '
                     f'({"STABLE ✓" if pred_orig == pred_smooth else "CHANGED ✗"})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        # Row 2, Panel 1: Top DROPS (concepts suppressed by smoothing)
        ax = axes[1, 0]
        if m_drops:
            drop_names = [d[0][:22] for d in m_drops[:10]]
            drop_orig = [d[2] for d in m_drops[:10]]
            drop_smooth = [d[3] for d in m_drops[:10]]
            y_d = np.arange(len(drop_names))
            ax.barh(y_d - 0.2, drop_orig, height=0.35, color='steelblue', label='Original', alpha=0.8)
            ax.barh(y_d + 0.2, drop_smooth, height=0.35, color='#d32f2f', label='After smoothing', alpha=0.8)
            ax.set_yticks(y_d); ax.set_yticklabels(drop_names, fontsize=8)
            ax.invert_yaxis()
        ax.set_xlabel('Activation')
        ax.set_title('Manifold: Top Drops (suppressed)', fontsize=11, fontweight='bold', color='#d32f2f')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        # Row 2, Panel 2: Least activated (originally active, weakest after smoothing)
        ax = axes[1, 1]
        if m_least:
            least_names = [l[0][:22] for l in m_least[:10]]
            least_orig = [l[2] for l in m_least[:10]]
            least_smooth = [l[1] for l in m_least[:10]]
            y_l = np.arange(len(least_names))
            ax.barh(y_l - 0.2, least_orig, height=0.35, color='steelblue', label='Original', alpha=0.8)
            ax.barh(y_l + 0.2, least_smooth, height=0.35, color='#757575', label='After smoothing', alpha=0.8)
            ax.set_yticks(y_l); ax.set_yticklabels(least_names, fontsize=8)
            ax.invert_yaxis()
        ax.set_xlabel('Activation')
        ax.set_title('Manifold: Least Activated (weakest survivors)', fontsize=11, fontweight='bold', color='#757575')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        # Row 2, Panel 3: empty
        axes[1, 2].axis('off')

        fig.suptitle(f'Manifold Smoothing — idx={target_idx}', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(manifold_dir, f"idx{target_idx}_manifold.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Manifold JSON
        manifold_result = {
            'idx': target_idx,
            'image_path': img_path,
            'true_class': get_class_name(PROBE_DATASET, label_true),
            'label_true': label_true,
            'method': 'manifold',
            'pred_orig': pred_orig,
            'pred_orig_class': get_class_name(PROBE_DATASET, pred_orig),
            'pred_smooth': pred_smooth,
            'pred_smooth_class': get_class_name(PROBE_DATASET, pred_smooth),
            'n_votes': n_votes,
            'stable': bool(pred_orig == pred_smooth),
            'mean_overlap': round(float(np.mean(smooth_concepts_overlap)), 4),
            'mean_concept_survival': mean_concept_survival,
            'n_concepts_certified': n_concepts_certified,
            'concept_survival': concept_survival_named,
            'params': {'K_NEIGHBORS': K_NEIGHBORS, 'SCALE_WEIGHT': SCALE_WEIGHT, 'N_SMOOTH_SAMPLES': N_SMOOTH_SAMPLES},
            'original_concepts': [{'name': n, 'value': round(v, 4)} for n, v in orig_concepts],
            'top5_neighbors': top5_neighbors_info,
            'example_noisy_samples': example_noisy_samples,
            'final_avg_concepts': [{'name': n, 'value': round(v, 4)} for n, v in final_concepts],
            'top_drops': [{'name': d[0], 'drop': d[1], 'orig': d[2], 'smoothed': d[3]} for d in m_drops[:10]],
            'top_gains': [{'name': g[0], 'gain': g[1], 'orig': g[2], 'smoothed': g[3]} for g in m_gains[:10]],
            'least_activated': [{'name': l[0], 'smoothed': l[1], 'orig': l[2]} for l in m_least[:10]],
        }
        with open(os.path.join(manifold_dir, f"idx{target_idx}_detail.json"), 'w') as f:
            json.dump(manifold_result, f, indent=2)

        # --- Isotropic visualization (2x3 grid) ---
        fig, axes = plt.subplots(2, 3, figsize=(26, 16))

        # Generate isotropic noisy points for visualization (project onto same PCA)
        vis_gauss_points = []
        vis_gauss_preds_list = []
        for _ in range(N_SMOOTH_SAMPLES):
            noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
            cv_g = cv_orig + noise
            vis_gauss_points.append(cv_g)
            p = classify_concept_vector(
                torch.tensor(cv_g, dtype=torch.float32, device=args.device),
                classifier_weights)
            vis_gauss_preds_list.append(p)
        gauss_vis = pca_vis.transform(np.stack(vis_gauss_points))
        gauss_correct = [p == label_true for p in vis_gauss_preds_list]

        # Row 1, Panel 1: Neighborhood reference
        ax = axes[0, 0]
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightblue', s=8, alpha=0.4, label=f'{K_NEIGHBORS} KNN')
        ax.scatter(orig_vis[0], orig_vis[1], c='red', s=200, marker='*', zorder=10,
                   edgecolors='darkred', linewidths=1.5, label='Target')
        ax.set_title(f'Concept Space (PCA)\nTrue: {get_class_name(PROBE_DATASET, label_true)}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 1, Panel 2: Isotropic Gaussian noisy samples
        ax = axes[0, 1]
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightgray', s=5, alpha=0.2)
        colors_g = ['green' if c else 'orange' for c in gauss_correct]
        ax.scatter(gauss_vis[:, 0], gauss_vis[:, 1], c=colors_g, s=15, alpha=0.6,
                   label=f'{N_SMOOTH_SAMPLES} samples')
        ax.scatter(orig_vis[0], orig_vis[1], c='red', s=200, marker='*', zorder=10,
                   edgecolors='darkred', linewidths=1.5, label='Original')
        cov_g = np.cov(gauss_vis.T)
        ev_g, evec_g = np.linalg.eigh(cov_g)
        angle_g = np.degrees(np.arctan2(evec_g[1, 1], evec_g[0, 1]))
        for n_std in [1, 2]:
            ell = Ellipse(xy=gauss_vis.mean(axis=0), width=2*n_std*np.sqrt(ev_g[1]),
                          height=2*n_std*np.sqrt(ev_g[0]), angle=angle_g,
                          fill=False, edgecolor='purple', linestyle='--', linewidth=1.5, alpha=0.6)
            ax.add_patch(ell)
        n_correct_g = sum(gauss_correct)
        ax.set_title(f'Isotropic Gaussian (σ={gauss_sigma:.3f})\nCorrect: {n_correct_g}/{N_SMOOTH_SAMPLES}',
                     fontsize=11, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

        # Row 1, Panel 3: Isotropic concept bar chart (top activations)
        ax = axes[0, 2]
        gauss_top_concepts = get_top_concept_info(cv_gauss_avg, concept_names, top_k=n_show)
        all_names_g = []
        for n, _ in orig_top_concepts:
            if n not in all_names_g: all_names_g.append(n)
        for n, _ in gauss_top_concepts:
            if n not in all_names_g: all_names_g.append(n)
        all_names_g = all_names_g[:25]
        orig_vals_bar_g = [float(cv_orig[name_to_idx[n]]) if n in name_to_idx else 0 for n in all_names_g]
        smooth_vals_bar_g = [float(cv_gauss_avg[name_to_idx[n]]) if n in name_to_idx else 0 for n in all_names_g]
        y_pos_g = np.arange(len(all_names_g))
        ax.barh(y_pos_g - 0.2, orig_vals_bar_g, height=0.35, color='steelblue', label='Original', alpha=0.8)
        ax.barh(y_pos_g + 0.2, smooth_vals_bar_g, height=0.35, color='coral', label='Isotropic avg', alpha=0.8)
        ax.set_yticks(y_pos_g); ax.set_yticklabels([n[:22] for n in all_names_g], fontsize=8)
        ax.invert_yaxis(); ax.set_xlabel('Activation')
        ax.set_title(f'Top Activations: {get_class_name(PROBE_DATASET, pred_orig)} → '
                     f'{get_class_name(PROBE_DATASET, pred_gauss)} '
                     f'({"STABLE ✓" if pred_orig == pred_gauss else "CHANGED ✗"})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        # Row 2, Panel 1: Top DROPS
        ax = axes[1, 0]
        if g_drops:
            drop_names = [d[0][:22] for d in g_drops[:10]]
            drop_orig = [d[2] for d in g_drops[:10]]
            drop_smooth = [d[3] for d in g_drops[:10]]
            y_d = np.arange(len(drop_names))
            ax.barh(y_d - 0.2, drop_orig, height=0.35, color='steelblue', label='Original', alpha=0.8)
            ax.barh(y_d + 0.2, drop_smooth, height=0.35, color='#d32f2f', label='After smoothing', alpha=0.8)
            ax.set_yticks(y_d); ax.set_yticklabels(drop_names, fontsize=8)
            ax.invert_yaxis()
        ax.set_xlabel('Activation')
        ax.set_title('Isotropic: Top Drops (suppressed)', fontsize=11, fontweight='bold', color='#d32f2f')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        # Row 2, Panel 2: Least activated (originally active, weakest after smoothing)
        ax = axes[1, 1]
        if g_least:
            least_names = [l[0][:22] for l in g_least[:10]]
            least_orig = [l[2] for l in g_least[:10]]
            least_smooth = [l[1] for l in g_least[:10]]
            y_l = np.arange(len(least_names))
            ax.barh(y_l - 0.2, least_orig, height=0.35, color='steelblue', label='Original', alpha=0.8)
            ax.barh(y_l + 0.2, least_smooth, height=0.35, color='#757575', label='After smoothing', alpha=0.8)
            ax.set_yticks(y_l); ax.set_yticklabels(least_names, fontsize=8)
            ax.invert_yaxis()
        ax.set_xlabel('Activation')
        ax.set_title('Isotropic: Least Activated (weakest survivors)', fontsize=11, fontweight='bold', color='#757575')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        # Row 2, Panel 3: empty
        axes[1, 2].axis('off')

        fig.suptitle(f'Isotropic Gaussian Smoothing — idx={target_idx}', fontsize=14, fontweight='bold', y=1.01)
        plt.tight_layout()
        plt.savefig(os.path.join(isotropic_dir, f"idx{target_idx}_isotropic.png"), dpi=150, bbox_inches='tight')
        plt.close(fig)

        # Isotropic JSON
        isotropic_result = {
            'idx': target_idx,
            'image_path': img_path,
            'true_class': get_class_name(PROBE_DATASET, label_true),
            'label_true': label_true,
            'method': 'isotropic_gaussian',
            'pred_orig': pred_orig,
            'pred_orig_class': get_class_name(PROBE_DATASET, pred_orig),
            'pred_smooth': pred_gauss,
            'pred_smooth_class': get_class_name(PROBE_DATASET, pred_gauss),
            'n_votes': n_votes_gauss,
            'stable': bool(pred_orig == pred_gauss),
            'mean_overlap': round(float(np.mean(gauss_overlap)), 4),
            'mean_concept_survival': g_mean_concept_survival,
            'n_concepts_certified': g_n_concepts_certified,
            'concept_survival': gauss_concept_survival_named,
            'params': {'GAUSS_SIGMA': round(float(gauss_sigma), 4), 'N_SMOOTH_SAMPLES': N_SMOOTH_SAMPLES},
            'original_concepts': [{'name': n, 'value': round(v, 4)} for n, v in orig_concepts],
            'example_noisy_samples': gauss_example_samples,
            'final_avg_concepts': [{'name': n, 'value': round(v, 4)} for n, v in gauss_final_concepts],
            'top_drops': [{'name': d[0], 'drop': d[1], 'orig': d[2], 'smoothed': d[3]} for d in g_drops[:10]],
            'top_gains': [{'name': g[0], 'gain': g[1], 'orig': g[2], 'smoothed': g[3]} for g in g_gains[:10]],
            'least_activated': [{'name': l[0], 'smoothed': l[1], 'orig': l[2]} for l in g_least[:10]],
        }
        with open(os.path.join(isotropic_dir, f"idx{target_idx}_detail.json"), 'w') as f:
            json.dump(isotropic_result, f, indent=2)

        # Also save input image to both folders
        if img_path and os.path.exists(img_path):
            import shutil
            ext = os.path.splitext(img_path)[1]
            shutil.copy2(img_path, os.path.join(manifold_dir, f"idx{target_idx}_input{ext}"))
            shutil.copy2(img_path, os.path.join(isotropic_dir, f"idx{target_idx}_input{ext}"))

        # ---------------------------------------------------------------
        # Concept Vote Histogram — Manifold
        # Shows how often each concept appears in top-K across N samples
        # Original top-20 concepts highlighted in blue, new concepts in orange
        # ---------------------------------------------------------------
        def _plot_concept_vote_histogram(concept_votes, orig_top_set, concept_names_list,
                                         n_samples, method_name, color_orig, color_new,
                                         save_path, target_idx, n_show=50):
            """Bar chart: x=concepts (union of top across N samples), y=votes (how many samples)."""
            # Get top concepts by vote count, limit to n_show
            top_concepts = concept_votes.most_common(n_show)
            if not top_concepts:
                return

            c_idxs = [c[0] for c in top_concepts]
            c_votes = [c[1] for c in top_concepts]
            c_names = [(concept_names_list[i] if concept_names_list else f"c_{i}")[:25]
                       for i in c_idxs]
            c_colors = [color_orig if i in orig_top_set else color_new for i in c_idxs]

            fig, ax = plt.subplots(figsize=(max(14, len(c_names) * 0.35), 6))
            bars = ax.bar(range(len(c_names)), c_votes, color=c_colors, alpha=0.85, edgecolor='white', linewidth=0.5)
            ax.set_xticks(range(len(c_names)))
            ax.set_xticklabels(c_names, rotation=60, ha='right', fontsize=7)
            ax.set_ylabel(f'Votes (out of {n_samples})')
            ax.set_xlabel('Concept')
            ax.axhline(y=n_samples * 0.5, color='red', ls='--', alpha=0.5, label='50% threshold')

            # Legend
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor=color_orig, label=f'Original top-{orig_top_k} (survived)'),
                Patch(facecolor=color_new, label='New concepts (appeared after smoothing)'),
            ]
            ax.legend(handles=legend_elements, fontsize=9, loc='upper right')

            ax.set_title(f'{method_name} — Concept Votes across {n_samples} smooth samples\n'
                         f'idx={target_idx}, true={get_class_name(PROBE_DATASET, label_true)}',
                         fontsize=12, fontweight='bold')
            ax.set_ylim(0, n_samples * 1.08)
            ax.grid(True, axis='y', alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        _plot_concept_vote_histogram(
            all_concept_votes_manifold, orig_top, concept_names,
            N_SMOOTH_SAMPLES, 'Manifold Smoothing', '#2196F3', '#FF9800',
            os.path.join(manifold_dir, f"idx{target_idx}_concept_votes.png"),
            target_idx)

        _plot_concept_vote_histogram(
            all_concept_votes_gaussian, orig_top, concept_names,
            N_SMOOTH_SAMPLES, 'Isotropic Gaussian', '#2196F3', '#FF9800',
            os.path.join(isotropic_dir, f"idx{target_idx}_concept_votes.png"),
            target_idx)

        # ---------------------------------------------------------------
        # Least Activated Concepts — standalone bar chart (like top activations)
        # Shows originally-active concepts with lowest activation values,
        # with concept index shown, comparing original vs smoothed.
        # ---------------------------------------------------------------
        def _plot_least_activated(cv_orig, cv_smoothed_avg, concept_names_list,
                                  method_name, color_smooth, save_path, target_idx,
                                  n_show=20):
            """Bar chart of least-activated originally-active concepts, with concept index."""
            # Find all originally active concepts (activation > 1e-6)
            active_mask = cv_orig > 1e-6
            if active_mask.sum() == 0:
                return
            active_idxs = np.where(active_mask)[0]
            # Sort by original activation (ascending = least first)
            sorted_by_act = sorted(active_idxs, key=lambda i: cv_orig[i])[:n_show]

            c_labels = []
            for i in sorted_by_act:
                cname = (concept_names_list[i] if concept_names_list else f"c_{i}")[:22]
                c_labels.append(f"[{i}] {cname}")

            orig_vals = [float(cv_orig[i]) for i in sorted_by_act]
            smooth_vals = [float(cv_smoothed_avg[i]) for i in sorted_by_act]

            fig, ax = plt.subplots(figsize=(12, max(6, len(c_labels) * 0.4)))
            y_pos = np.arange(len(c_labels))
            ax.barh(y_pos - 0.2, orig_vals, height=0.35, color='steelblue',
                    label='Original', alpha=0.8)
            ax.barh(y_pos + 0.2, smooth_vals, height=0.35, color=color_smooth,
                    label=f'{method_name} avg', alpha=0.8)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(c_labels, fontsize=8)
            ax.invert_yaxis()
            ax.set_xlabel('Activation')
            ax.set_title(f'{method_name}: Least Activated Concepts (sorted by original activation)\n'
                         f'idx={target_idx}, true={get_class_name(PROBE_DATASET, label_true)}',
                         fontsize=12, fontweight='bold')
            ax.legend(fontsize=9)
            ax.grid(True, axis='x', alpha=0.3)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        _plot_least_activated(
            cv_orig, cv_smoothed_avg, concept_names,
            'Manifold', 'coral',
            os.path.join(manifold_dir, f"idx{target_idx}_least_activated.png"),
            target_idx)

        _plot_least_activated(
            cv_orig, cv_gauss_avg, concept_names,
            'Gaussian', '#FF9800',
            os.path.join(isotropic_dir, f"idx{target_idx}_least_activated.png"),
            target_idx)

        # ---------------------------------------------------------------
        # Concept Activation Distribution — boxplots showing spread across N samples
        # One figure per method: for each original top-K concept, show the
        # distribution of its activation value across all N smooth samples,
        # with the original activation marked as a red diamond.
        # ---------------------------------------------------------------
        def _plot_activation_distribution(all_smooth_vecs, orig_top_idxs_arr, cv_orig,
                                          concept_votes, concept_names_list, orig_top_set,
                                          method_name, color_top, color_least, color_new,
                                          save_path, target_idx, n_samples, n_new=10):
            """3-row box plot: top-20 (original), least-activated (original near-zero), new concepts.
            
            Row 1: Original top-K concepts sorted by original activation
            Row 2: Least activated — originally active concepts with lowest activation values
            Row 3: New concepts — NOT in original top-K but appeared frequently after smoothing
            """
            smooth_mat = np.stack(all_smooth_vecs)  # (N, 8192)

            # --- Row 1: Original top-K sorted by original activation ---
            sorted_top = sorted(orig_top_idxs_arr, key=lambda i: cv_orig[i], reverse=True)

            # --- Row 2: Least activated (originally had low but non-zero activation) ---
            # Find originally-active concepts (activation > 1e-6) NOT in top-K, sorted by
            # original activation ascending. These are on the boundary.
            all_active = np.where(cv_orig > 1e-6)[0]
            least_candidates = [i for i in all_active if i not in orig_top_set]
            least_candidates = sorted(least_candidates, key=lambda i: cv_orig[i])[:n_new]
            # If not enough non-top-K active concepts, include the weakest from top-K
            if len(least_candidates) < n_new:
                weak_top = sorted(orig_top_idxs_arr, key=lambda i: cv_orig[i])[:n_new - len(least_candidates)]
                least_candidates = least_candidates + [i for i in weak_top if i not in least_candidates]

            # --- Row 3: New concepts (NOT in original top-K, appeared in smooth samples) ---
            new_concepts = [(cidx, cnt) for cidx, cnt in concept_votes.most_common()
                          if cidx not in orig_top_set]
            new_concept_idxs = [c[0] for c in new_concepts[:n_new]]

            # Helper to make one row of box plots
            def _draw_row(ax, concept_idxs, row_color, row_label):
                if not concept_idxs:
                    ax.text(0.5, 0.5, 'None', ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(row_label, fontsize=11, fontweight='bold')
                    return
                c_names = [(concept_names_list[i] if concept_names_list else f"c_{i}")[:25]
                           for i in concept_idxs]
                data = [smooth_mat[:, i].tolist() for i in concept_idxs]
                orig_vals = [float(cv_orig[i]) for i in concept_idxs]

                bp = ax.boxplot(data, positions=range(len(c_names)), widths=0.6,
                               patch_artist=True, showfliers=False,
                               medianprops=dict(color='black', linewidth=1.5))
                for patch in bp['boxes']:
                    patch.set_facecolor(row_color)
                    patch.set_alpha(0.6)
                ax.scatter(range(len(c_names)), orig_vals, c='red', s=80, marker='D',
                          zorder=5, edgecolors='darkred', linewidths=1, label='Original')
                for j, d in enumerate(data):
                    jitter = np.random.normal(0, 0.08, size=len(d))
                    ax.scatter(np.full(len(d), j) + jitter, d, c=row_color, s=3, alpha=0.15, zorder=2)
                ax.set_xticks(range(len(c_names)))
                ax.set_xticklabels(c_names, rotation=45, ha='right', fontsize=8)
                ax.set_ylabel('Activation')
                ax.set_title(row_label, fontsize=11, fontweight='bold')
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, axis='y', alpha=0.3)
                ax.axhline(y=0, color='gray', ls='-', alpha=0.3)

            fig, axes = plt.subplots(3, 1, figsize=(max(14, max(len(sorted_top), n_new) * 0.6), 18))

            _draw_row(axes[0], sorted_top, color_top,
                      f'Top-{orig_top_k} Original Concepts (sorted by activation)')
            _draw_row(axes[1], least_candidates, color_least,
                      f'Least Activated (weak/boundary concepts)')
            _draw_row(axes[2], new_concept_idxs, color_new,
                      f'Top-{n_new} NEW Concepts (not in original top-{orig_top_k}, emerged after smoothing)')

            fig.suptitle(f'{method_name} — Activation Distributions (N={n_samples})\n'
                         f'idx={target_idx}, true={get_class_name(PROBE_DATASET, label_true)}',
                         fontsize=13, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        _plot_activation_distribution(
            all_manifold_smooth_vecs, orig_top_idxs, cv_orig,
            all_concept_votes_manifold, concept_names, orig_top,
            'Manifold Smoothing', '#2196F3', '#757575', '#FF9800',
            os.path.join(manifold_dir, f"idx{target_idx}_activation_dist.png"),
            target_idx, N_SMOOTH_SAMPLES)

        _plot_activation_distribution(
            all_gauss_smooth_vecs, orig_top_idxs, cv_orig,
            all_concept_votes_gaussian, concept_names, orig_top,
            'Isotropic Gaussian', '#2196F3', '#757575', '#FF9800',
            os.path.join(isotropic_dir, f"idx{target_idx}_activation_dist.png"),
            target_idx, N_SMOOTH_SAMPLES)

        print(f"\n  Saved viz+detail for idx {target_idx} to: {manifold_dir} & {isotropic_dir}")
        viz_count += 1
    else:
        if (loop_i + 1) % 50 == 0:
            print(f"  [{loop_i+1}/{N_TARGETS}] done")

    # --- Write result immediately to JSONL (survives OOM) ---
    # Paper-ready concept stability scores
    scores_manifold = compute_concept_scores(cv_orig, cv_smoothed_avg)
    scores_gaussian = compute_concept_scores(cv_orig, cv_gauss_avg)

    # Volume framework (Jonas): r_iso from gaussian, r_mani from manifold
    r_iso = gauss_class_cert['certified_radius']
    r_mani = class_cert['certified_radius']
    # Filter eigenvalues to non-trivial components
    ev_positive = ev[ev > 1e-10]
    CONCEPT_DIM = len(cv_orig)
    volumes = compute_volumes(r_iso, r_mani, ev_positive, D=CONCEPT_DIM)

    result_row = {
        'idx': target_idx,
        'img_path': str(img_path) if img_path else None,
        'label_true': label_true,
        'pred_orig': pred_orig,
        # === Concept-level certification (manifold) ===
        'mean_concept_survival_manifold': mean_concept_survival,
        'n_concepts_certified_manifold': n_concepts_certified,
        'min_concept_radius_manifold': min_concept_radius,
        'mean_concept_radius_manifold': mean_concept_radius,
        'concept_survival_manifold': concept_survival_named,
        'overlap_manifold': round(float(np.mean(smooth_concepts_overlap)), 4),
        # === Downstream class certification (manifold) ===
        'pred_manifold': pred_smooth,
        'class_cert_manifold': class_cert,
        'stable_manifold': pred_orig == pred_smooth and not class_cert['abstained'],
        # === Concept stability scores (manifold) ===
        'scores_manifold': scores_manifold,
        # === Concept-level certification (gaussian) ===
        'mean_concept_survival_gaussian': g_mean_concept_survival,
        'n_concepts_certified_gaussian': g_n_concepts_certified,
        'min_concept_radius_gaussian': g_min_concept_radius,
        'mean_concept_radius_gaussian': g_mean_concept_radius,
        'concept_survival_gaussian': gauss_concept_survival_named,
        'overlap_gaussian': round(float(np.mean(gauss_overlap)), 4),
        # === Downstream class certification (gaussian) ===
        'pred_gaussian': pred_gauss,
        'class_cert_gaussian': gauss_class_cert,
        'stable_gaussian': pred_orig == pred_gauss and not gauss_class_cert['abstained'],
        # === Concept stability scores (gaussian) ===
        'scores_gaussian': scores_gaussian,
        # === Volume framework ===
        'volumes': volumes,
        'n_eigenvalues': len(ev_positive),
        'top_eigenvalues': [round(float(e), 6) for e in ev_positive[:10]],
    }
    results_jsonl_file.write(json.dumps(result_row) + '\n')
    results_jsonl_file.flush()

results_jsonl_file.close()

# ===========================================================================
# Summary — read back from JSONL (works even after resume)
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

results = []
with open(results_jsonl_path, 'r') as f:
    for line in f:
        line = line.strip()
        if line:
            try:
                results.append(json.loads(line))
            except json.JSONDecodeError:
                pass

print(f"Total results: {len(results)}")

# --- Concept-level certification ---
print(f"\n{'='*80}")
print(f"CONCEPT CERTIFICATION (top-{orig_top_k} concepts, α={CERTIFY_ALPHA})")
print(f"{'='*80}")
m_surv = [r.get('mean_concept_survival_manifold', 0) for r in results]
g_surv = [r.get('mean_concept_survival_gaussian', 0) for r in results]
m_cert = [r.get('n_concepts_certified_manifold', 0) for r in results]
g_cert = [r.get('n_concepts_certified_gaussian', 0) for r in results]
m_min_r = [r.get('min_concept_radius_manifold', 0) for r in results]
g_min_r = [r.get('min_concept_radius_gaussian', 0) for r in results]
m_mean_r = [r.get('mean_concept_radius_manifold', 0) for r in results]
g_mean_r = [r.get('mean_concept_radius_gaussian', 0) for r in results]
m_overlap = [r.get('overlap_manifold', 0) for r in results]
g_overlap = [r.get('overlap_gaussian', 0) for r in results]

header = f"{'Method':<12s} {'Survival':<10s} {'#Cert/'+str(orig_top_k):<10s} {'Overlap':<10s} {'MinRadius':<12s} {'MeanRadius':<12s}"
print(header)
print(f"{'-'*66}")
print(f"{'Manifold':<12s} {np.mean(m_surv):<10.3f} {np.mean(m_cert):<10.1f} {np.mean(m_overlap):<10.3f} "
      f"{np.mean(m_min_r):<12.4f} {np.mean(m_mean_r):<12.4f}")
print(f"{'Gaussian':<12s} {np.mean(g_surv):<10.3f} {np.mean(g_cert):<10.1f} {np.mean(g_overlap):<10.3f} "
      f"{np.mean(g_min_r):<12.4f} {np.mean(g_mean_r):<12.4f}")

# --- Downstream class certification ---
print(f"\n{'='*80}")
print(f"DOWNSTREAM CLASS CERTIFICATION (Cohen et al., α={CERTIFY_ALPHA})")
print(f"{'='*80}")

def _class_stats(results, method):
    """Extract class certification stats."""
    cert_key = f'class_cert_{method}'
    stable_key = f'stable_{method}'
    pred_key = f'pred_{method}'

    n_correct_certified = 0
    n_abstained = 0
    radii = []
    for r in results:
        cc = r.get(cert_key, {})
        if cc.get('abstained', True):
            n_abstained += 1
        else:
            if r[pred_key] == r['pred_orig']:  # prediction matches original
                n_correct_certified += 1
            radii.append(cc.get('certified_radius', 0))
    n = len(results)
    return {
        'certified_accuracy': n_correct_certified / n if n else 0,
        'abstain_rate': n_abstained / n if n else 0,
        'n_certified': n - n_abstained,
        'n_correct_certified': n_correct_certified,
        'mean_radius': np.mean(radii) if radii else 0,
        'median_radius': np.median(radii) if radii else 0,
        'max_radius': max(radii) if radii else 0,
    }

m_cls = _class_stats(results, 'manifold')
g_cls = _class_stats(results, 'gaussian')
n = len(results)

header2 = f"{'Method':<12s} {'CertAcc':<10s} {'Abstain':<10s} {'#Cert':<8s} {'MeanR':<10s} {'MedianR':<10s} {'MaxR':<10s}"
print(header2)
print(f"{'-'*70}")
print(f"{'Manifold':<12s} {m_cls['certified_accuracy']:<10.3f} {m_cls['abstain_rate']:<10.3f} "
      f"{str(m_cls['n_certified'])+'/'+str(n):<8s} "
      f"{m_cls['mean_radius']:<10.4f} {m_cls['median_radius']:<10.4f} {m_cls['max_radius']:<10.4f}")
print(f"{'Gaussian':<12s} {g_cls['certified_accuracy']:<10.3f} {g_cls['abstain_rate']:<10.3f} "
      f"{str(g_cls['n_certified'])+'/'+str(n):<8s} "
      f"{g_cls['mean_radius']:<10.4f} {g_cls['median_radius']:<10.4f} {g_cls['max_radius']:<10.4f}")

# --- Concept Stability Scores (paper-ready) ---
print(f"\n{'='*80}")
print("CONCEPT STABILITY SCORES (paper-ready)")
print(f"{'='*80}")
score_keys = ['concept_fidelity', 'rank_correlation', 'concept_drift',
              'spurious_act_rate', 'act_drop_score', 'act_gain_score']
score_labels = ['Fidelity(r)', 'RankCorr(τ)', 'Drift(L2)', 'SAR', 'ADS', 'AGS']
header3 = f"{'Method':<12s} " + " ".join(f"{l:<12s}" for l in score_labels)
print(header3)
print(f"{'-'*86}")
for method, label in [('manifold', 'Manifold'), ('gaussian', 'Gaussian')]:
    vals = []
    for k in score_keys:
        v = [r.get(f'scores_{method}', {}).get(k, 0) for r in results]
        vals.append(np.mean(v))
    print(f"{label:<12s} " + " ".join(f"{v:<12.4f}" for v in vals))

# --- Volume framework ---
print(f"\n{'='*80}")
print("CERTIFIED VOLUME (log-space)")
print(f"{'='*80}")
vol_keys = ['log_vol_iso_D', 'log_vol_iso_k', 'log_vol_mani_pred', 'log_vol_mani_actual']
vol_labels = ['Qty1(IsoD)', 'Qty2(Isok)', 'Qty3(ManiPred)', 'Qty4(ManiActual)']
header4 = f"{'Metric':<18s} {'Mean':<12s} {'Median':<12s} {'Std':<12s}"
print(header4)
print(f"{'-'*54}")
for vk, vl in zip(vol_keys, vol_labels):
    vals = [r.get('volumes', {}).get(vk, -np.inf) for r in results]
    vals_finite = [v for v in vals if v > -1e30]
    if vals_finite:
        print(f"{vl:<18s} {np.mean(vals_finite):<12.2f} {np.median(vals_finite):<12.2f} {np.std(vals_finite):<12.2f}")
    else:
        print(f"{vl:<18s} {'N/A':<12s} {'N/A':<12s} {'N/A':<12s}")
mean_k = np.mean([r.get('n_eigenvalues', 0) for r in results])
print(f"\nEffective manifold dim (mean): k={mean_k:.1f} / D={len(cv_orig)}")

# Write final JSON summaries
results_path = os.path.join(SAVE_DIR, f"smoothing_results_{PROBE_DATASET}.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

manifold_summary = [{'idx': r['idx'], 'label_true': r['label_true'], 'pred_orig': r['pred_orig'],
                     'pred_smooth': r['pred_manifold'], 'class_cert': r.get('class_cert_manifold', {}),
                     'stable_class': r['stable_manifold'], 'mean_overlap': r['overlap_manifold'],
                     'mean_concept_survival': r.get('mean_concept_survival_manifold', None),
                     'n_concepts_certified': r.get('n_concepts_certified_manifold', None),
                     'min_concept_radius': r.get('min_concept_radius_manifold', None),
                     'mean_concept_radius': r.get('mean_concept_radius_manifold', None),
                     'scores': r.get('scores_manifold', {})} for r in results]
isotropic_summary = [{'idx': r['idx'], 'label_true': r['label_true'], 'pred_orig': r['pred_orig'],
                      'pred_smooth': r['pred_gaussian'], 'class_cert': r.get('class_cert_gaussian', {}),
                      'stable_class': r['stable_gaussian'], 'mean_overlap': r['overlap_gaussian'],
                      'mean_concept_survival': r.get('mean_concept_survival_gaussian', None),
                      'n_concepts_certified': r.get('n_concepts_certified_gaussian', None),
                      'min_concept_radius': r.get('min_concept_radius_gaussian', None),
                      'mean_concept_radius': r.get('mean_concept_radius_gaussian', None),
                      'scores': r.get('scores_gaussian', {})} for r in results]
with open(os.path.join(manifold_dir, "summary.json"), 'w') as f:
    json.dump(manifold_summary, f, indent=2)
with open(os.path.join(isotropic_dir, "summary.json"), 'w') as f:
    json.dump(isotropic_summary, f, indent=2)

print(f"\nResults saved to:")
print(f"  JSONL (incremental): {results_jsonl_path}")
print(f"  Combined JSON:       {results_path}")
print(f"  Manifold summary:    {manifold_dir}/summary.json")
print(f"  Isotropic summary:   {isotropic_dir}/summary.json")
