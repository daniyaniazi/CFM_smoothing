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
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')  # no display on server
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from cfm.arg_parser import get_default_parser
from cfm.utils import common_init, get_img_model, get_probe_dataset
from cfm.cfm import CFM
from cfm import config as cfm_config
from cfm.method_utils import MethodCFM
from cfm.data_utils import probe_classnames
from dictionary_learning.utils import load_dictionary

# ===========================================================================
# Config
# ===========================================================================
CONFIG_NAME = 'k_12_ef_16_lr_0.0001_mf_[0.008,0.03,0.06,0.12,0.24,0.542]'
PROBE_DATASET = "imagenet"       # or "places365"
PROBE_SPLIT = "val"
PROBE_CONFIG = "lr0.0001_bs512_epo50_clCE_spL1_spl0.0max_no_threshold"

# Smoothing parameters
K_NEIGHBORS = 500
SCALE_WEIGHT = 0.7
N_SMOOTH_SAMPLES = 100
N_TARGETS = 500           # number of val images to certify
N_VIZ = 10                # save detailed viz/JSON only for the first N images

# Data generation
BATCH_SIZE = 64
NUM_WORKERS = 4

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'smoothing_data')
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# Helpers
# ===========================================================================
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


# ===========================================================================
# Step 1: Load CFM model
# ===========================================================================
print("=" * 70)
print("Loading CFM model")
print("=" * 70)

parser = get_default_parser()
args = parser.parse_args([])
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
common_init(args)
args.config_name = CONFIG_NAME

print(f"Device: {args.device}")

feature_extractor, preprocess = get_img_model(args)
feature_extractor.eval()
print("CLIP-DINOiser loaded")

sae_base = Path(args.save_dir_sae_ckpts['img']) / args.config_name / 'trainer_0'
assert sae_base.exists(), f"SAE not found at {sae_base}"
autoencoder, ae_config = load_dictionary(str(sae_base), args.device)
print("SAE loaded")

cfm_model = CFM(
    feature_extractor=feature_extractor,
    autoencoder=autoencoder,
    apply_found=False,
    device=args.device,
)
cfm_model.eval()
print("CFM model ready")

# Concept names (optional)
# Try override path first (avoids bracket issues in Kai's SAE path)
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


# ===========================================================================
# Step 2: Load dataset + linear probe
# ===========================================================================
print("\n" + "=" * 70)
print("Loading datasets and linear classifier")
print("=" * 70)

args.probe_dataset = PROBE_DATASET
args.probe_split = PROBE_SPLIT
args.probe_dataset_root_dir = cfm_config.probe_dataset_root_dir_dict[PROBE_DATASET]

# Load TRAIN dataset (for KNN index / manifold)
probe_train_dataset = get_probe_dataset(
    PROBE_DATASET, "train", args.probe_dataset_root_dir, preprocess_fn=preprocess)
print(f"Train dataset: {PROBE_DATASET}, {len(probe_train_dataset)} samples")

# Load VAL dataset (for testing)
probe_val_dataset = get_probe_dataset(
    PROBE_DATASET, PROBE_SPLIT, args.probe_dataset_root_dir, preprocess_fn=preprocess)
print(f"Val dataset: {PROBE_DATASET} ({PROBE_SPLIT}), {len(probe_val_dataset)} samples")

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
# Step 3: Generate concept vectors for TRAIN (for KNN index) and VAL (for testing)
# ===========================================================================

def generate_concept_vectors(dataset, split_name, save_dir, cfm_model, device, batch_size, num_workers):
    cv_path = os.path.join(save_dir, f"concept_vectors_{PROBE_DATASET}_{split_name}.pt")
    labels_path = os.path.join(save_dir, f"labels_{PROBE_DATASET}_{split_name}.pt")

    if os.path.exists(cv_path) and os.path.exists(labels_path):
        print(f"\nLoading cached {split_name} concept vectors")
        vectors = torch.load(cv_path)
        labels = torch.load(labels_path)
        print(f"Loaded {vectors.shape[0]} {split_name} vectors from {cv_path}")
    else:
        print(f"\nGenerating {split_name} concept vectors (this takes a while)", flush=True)
        loader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, num_workers=num_workers, pin_memory=True)
        all_vecs = []
        all_labs = []
        import time as _time
        t0 = _time.time()
        with torch.no_grad():
            for batch_idx, (imgs, labs) in enumerate(loader):
                imgs = imgs.to(device)
                cvs = cfm_model.get_aggregated_concept_activations(imgs)
                all_vecs.append(cvs.cpu())
                all_labs.append(labs)
                if (batch_idx + 1) % 100 == 0:
                    elapsed = _time.time() - t0
                    eta = elapsed / (batch_idx + 1) * (len(loader) - batch_idx - 1)
                    print(f"  {split_name} batch {batch_idx+1}/{len(loader)}  "
                          f"elapsed {elapsed/60:.1f}min  ETA {eta/60:.1f}min", flush=True)
        vectors = torch.cat(all_vecs, dim=0)
        labels = torch.cat(all_labs, dim=0)
        torch.save(vectors, cv_path)
        torch.save(labels, labels_path)
        print(f"Saved {vectors.shape[0]} {split_name} concept vectors to {cv_path}")

    print(f"{split_name} shape: {vectors.shape}, non-zero/vec (mean): {(vectors > 0).sum(1).float().mean():.1f}")
    return vectors, labels

# Train vectors (for KNN index / manifold)
train_concept_vectors, train_labels = generate_concept_vectors(
    probe_train_dataset, "train", SAVE_DIR, cfm_model, args.device, BATCH_SIZE, NUM_WORKERS)

# Val vectors (for testing targets)
val_concept_vectors, val_labels = generate_concept_vectors(
    probe_val_dataset, "val", SAVE_DIR, cfm_model, args.device, BATCH_SIZE, NUM_WORKERS)


# ===========================================================================
# Step 4: Subsample train + build Annoy KNN index (cached)
# ===========================================================================
print("\n" + "=" * 70)
print("Building KNN index on TRAIN concept vectors")
print("=" * 70)

import annoy

N_TRAIN_SUBSAMPLE = 200_000  # subsample to avoid OOM
N_TRAIN_FULL = train_concept_vectors.shape[0]

if N_TRAIN_FULL > N_TRAIN_SUBSAMPLE:
    np.random.seed(123)
    subsample_idcs = np.random.choice(N_TRAIN_FULL, size=N_TRAIN_SUBSAMPLE, replace=False)
    train_concept_vectors = train_concept_vectors[subsample_idcs]
    train_labels = train_labels[subsample_idcs]
    print(f"Subsampled train: {N_TRAIN_FULL} -> {N_TRAIN_SUBSAMPLE}", flush=True)

N_TRAIN = train_concept_vectors.shape[0]
CONCEPT_DIM = train_concept_vectors.shape[1]

index_path = os.path.join(SAVE_DIR, f"knn_concepts_{PROBE_DATASET}_train_{N_TRAIN}.ann")

if os.path.exists(index_path):
    knn_index = annoy.AnnoyIndex(CONCEPT_DIM, 'euclidean')
    knn_index.load(index_path)
    print(f"Loaded existing KNN index from {index_path}", flush=True)
else:
    import time as _time
    knn_index = annoy.AnnoyIndex(CONCEPT_DIM, 'euclidean')
    print(f"Adding {N_TRAIN} items to index...", flush=True)
    t0 = _time.time()
    for i in range(N_TRAIN):
        knn_index.add_item(i, train_concept_vectors[i].numpy())
        if (i + 1) % 50000 == 0:
            print(f"  added {i+1}/{N_TRAIN} items ({_time.time()-t0:.0f}s)", flush=True)
    N_TREES = 10
    print(f"Building index with {N_TREES} trees...", flush=True)
    knn_index.build(N_TREES)
    knn_index.save(index_path)
    print(f"Built Annoy index: {N_TRAIN} vectors, dim={CONCEPT_DIM}, trees={N_TREES} "
          f"in {(_time.time()-t0)/60:.1f}min", flush=True)


# ===========================================================================
# Step 5: Manifold smoothing
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("Manifold smoothing", flush=True)
print("=" * 70, flush=True)
print(f"K={K_NEIGHBORS}, sigma={SCALE_WEIGHT}, N_samples={N_SMOOTH_SAMPLES}", flush=True)
print(f"Index: TRAIN ({N_TRAIN} vectors), Targets: VAL", flush=True)

results = []

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

    # Smoothing loop — collect 5 example noisy samples
    smooth_preds = []
    smooth_concepts_overlap = []
    example_noisy_samples = []
    orig_top = set(np.argsort(-cv_orig)[:20])

    for sample_i in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, SCALE_WEIGHT, size=len(ev))
        cv_noised = cv_whitened + noise
        cv_smoothed = cv_noised @ (np.sqrt(ev)[:, None] * Vt) + mean_nn

        pred_i = classify_concept_vector(
            torch.tensor(cv_smoothed, dtype=torch.float32, device=args.device),
            classifier_weights)
        smooth_preds.append(pred_i)

        noised_top = set(np.argsort(-cv_smoothed)[:20])
        overlap = len(orig_top & noised_top) / 20.0
        smooth_concepts_overlap.append(overlap)

        # Save first 5 noisy samples as examples
        if sample_i < 5:
            sample_concepts = get_top_concept_info(cv_smoothed, concept_names, top_k=20)
            example_noisy_samples.append({
                'sample_idx': sample_i + 1,
                'pred_class': get_class_name(PROBE_DATASET, pred_i),
                'top20_overlap': round(overlap, 3),
                'top_concepts': [{'name': n, 'value': round(v, 4)} for n, v in sample_concepts],
            })

    # Majority vote
    vote_counts = Counter(smooth_preds)
    pred_smooth, n_votes = vote_counts.most_common(1)[0]

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

    for sample_i in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
        cv_gauss = cv_orig + noise
        pred_g = classify_concept_vector(
            torch.tensor(cv_gauss, dtype=torch.float32, device=args.device),
            classifier_weights)
        gauss_preds.append(pred_g)
        gauss_top = set(np.argsort(-cv_gauss)[:20])
        g_overlap = len(orig_top & gauss_top) / 20.0
        gauss_overlap.append(g_overlap)
        if sample_i < 5:
            g_concepts = get_top_concept_info(cv_gauss, concept_names, top_k=20)
            gauss_example_samples.append({
                'sample_idx': sample_i + 1,
                'pred_class': get_class_name(PROBE_DATASET, pred_g),
                'top20_overlap': round(g_overlap, 3),
                'top_concepts': [{'name': n, 'value': round(v, 4)} for n, v in g_concepts],
            })

    gauss_vote_counts = Counter(gauss_preds)
    pred_gauss, n_votes_gauss = gauss_vote_counts.most_common(1)[0]

    cv_gauss_accum = np.zeros_like(cv_orig)
    for _ in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
        cv_gauss_accum += cv_orig + noise
    cv_gauss_avg = cv_gauss_accum / N_SMOOTH_SAMPLES
    gauss_final_concepts = get_top_concept_info(cv_gauss_avg, concept_names, top_k=20)

    # --- Print ---
    orig_concepts = get_top_concept_info(cv_orig, concept_names, top_k=20)
    save_viz = (viz_count < N_VIZ)

    print(f"\n[{loop_i+1}/{N_TARGETS}] idx={target_idx}  "
          f"true={get_class_name(PROBE_DATASET, label_true)}  "
          f"orig={get_class_name(PROBE_DATASET, pred_orig)}  "
          f"manifold={get_class_name(PROBE_DATASET, pred_smooth)}({n_votes}/{N_SMOOTH_SAMPLES})  "
          f"gauss={get_class_name(PROBE_DATASET, pred_gauss)}({n_votes_gauss}/{N_SMOOTH_SAMPLES})  "
          f"m_stable={pred_orig == pred_smooth}  g_stable={pred_orig == pred_gauss}", flush=True)

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
        # --- Compute shared x-axis limit for bar charts ---
        gauss_top_concepts_pre = get_top_concept_info(cv_gauss_avg, concept_names, top_k=20)
        all_bar_values = (
            [v for _, v in orig_concepts]
            + [v for _, v in final_concepts]
            + [v for _, v in gauss_top_concepts_pre]
        )
        shared_xlim = max(all_bar_values) * 1.1 if all_bar_values else 1.0

        # --- Manifold visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

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

        # Panel 1: Manifold neighborhood
        ax = axes[0]
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightblue', s=8, alpha=0.4, label=f'{K_NEIGHBORS} KNN neighbors')
        for rank in range(min(5, len(nn_idcs))):
            nn_cv_vis = pca_vis.transform(train_concept_vectors[nn_idcs[rank]].numpy().reshape(1, -1))[0]
            ax.scatter(nn_cv_vis[0], nn_cv_vis[1], c='blue', s=80, marker='D', zorder=5,
                       edgecolors='darkblue', linewidths=1.5)
            ax.annotate(f'N{rank+1}', (nn_cv_vis[0], nn_cv_vis[1]), fontsize=8, fontweight='bold',
                        xytext=(5, 5), textcoords='offset points')
        ax.scatter(orig_vis[0], orig_vis[1], c='red', s=200, marker='*', zorder=10,
                   edgecolors='darkred', linewidths=1.5, label='Target')
        ax.scatter(mean_vis[0], mean_vis[1], c='green', s=100, marker='X', zorder=10,
                   edgecolors='darkgreen', linewidths=1.5, label='Neighborhood mean')
        ax.set_title(f'Manifold Neighborhood (2D PCA)\nidx={target_idx}, True: {get_class_name(PROBE_DATASET, label_true)}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 2: Manifold noisy samples
        ax = axes[1]
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightgray', s=5, alpha=0.2)
        colors = ['green' if c else 'orange' for c in noisy_correct]
        ax.scatter(noisy_vis[:, 0], noisy_vis[:, 1], c=colors, s=15, alpha=0.6,
                   label=f'{N_SMOOTH_SAMPLES} manifold samples')
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
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 3: Manifold concept bar chart
        ax = axes[2]
        n_show = 20
        orig_top_concepts = get_top_concept_info(cv_orig, concept_names, top_k=n_show)
        final_top_concepts = get_top_concept_info(cv_smoothed_avg, concept_names, top_k=n_show)
        all_names = []
        for n, _ in orig_top_concepts:
            if n not in all_names: all_names.append(n)
        for n, _ in final_top_concepts:
            if n not in all_names: all_names.append(n)
        all_names = all_names[:30]
        orig_dict = {n: v for n, v in orig_top_concepts}
        final_dict = {n: v for n, v in final_top_concepts}
        y_pos = np.arange(len(all_names))
        ax.barh(y_pos - 0.2, [orig_dict.get(n, 0) for n in all_names], height=0.35, color='steelblue', label='Original', alpha=0.8)
        ax.barh(y_pos + 0.2, [final_dict.get(n, 0) for n in all_names], height=0.35, color='coral', label=f'Manifold avg', alpha=0.8)
        ax.set_yticks(y_pos); ax.set_yticklabels([n[:25] for n in all_names], fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel('Activation')
        ax.set_title(f'Manifold: {get_class_name(PROBE_DATASET, pred_orig)} → '
                     f'{get_class_name(PROBE_DATASET, pred_smooth)} '
                     f'({"STABLE ✓" if pred_orig == pred_smooth else "CHANGED ✗"})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(0, shared_xlim)
        ax.legend(fontsize=9); ax.grid(True, axis='x', alpha=0.3)
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
            'stable': pred_orig == pred_smooth,
            'mean_overlap': round(float(np.mean(smooth_concepts_overlap)), 4),
            'params': {'K_NEIGHBORS': K_NEIGHBORS, 'SCALE_WEIGHT': SCALE_WEIGHT, 'N_SMOOTH_SAMPLES': N_SMOOTH_SAMPLES},
            'original_concepts': [{'name': n, 'value': round(v, 4)} for n, v in orig_concepts],
            'top5_neighbors': top5_neighbors_info,
            'example_noisy_samples': example_noisy_samples,
            'final_avg_concepts': [{'name': n, 'value': round(v, 4)} for n, v in final_concepts],
        }
        with open(os.path.join(manifold_dir, f"idx{target_idx}_detail.json"), 'w') as f:
            json.dump(manifold_result, f, indent=2)

        # --- Isotropic visualization ---
        fig, axes = plt.subplots(1, 3, figsize=(24, 8))

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

        # Panel 1: Same neighborhood for reference
        ax = axes[0]
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightblue', s=8, alpha=0.4, label=f'{K_NEIGHBORS} KNN neighbors')
        ax.scatter(orig_vis[0], orig_vis[1], c='red', s=200, marker='*', zorder=10,
                   edgecolors='darkred', linewidths=1.5, label='Target')
        ax.set_title(f'Concept Space (2D PCA)\nidx={target_idx}, True: {get_class_name(PROBE_DATASET, label_true)}',
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 2: Isotropic Gaussian noisy samples
        ax = axes[1]
        ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightgray', s=5, alpha=0.2)
        colors_g = ['green' if c else 'orange' for c in gauss_correct]
        ax.scatter(gauss_vis[:, 0], gauss_vis[:, 1], c=colors_g, s=15, alpha=0.6,
                   label=f'{N_SMOOTH_SAMPLES} isotropic samples')
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
                     fontsize=12, fontweight='bold')
        ax.set_xlabel('PC1'); ax.set_ylabel('PC2'); ax.legend(fontsize=9); ax.grid(True, alpha=0.3)

        # Panel 3: Isotropic concept bar chart
        ax = axes[2]
        gauss_top_concepts = get_top_concept_info(cv_gauss_avg, concept_names, top_k=n_show)
        all_names_g = []
        for n, _ in orig_top_concepts:
            if n not in all_names_g: all_names_g.append(n)
        for n, _ in gauss_top_concepts:
            if n not in all_names_g: all_names_g.append(n)
        all_names_g = all_names_g[:30]
        orig_dict_g = {n: v for n, v in orig_top_concepts}
        gauss_dict = {n: v for n, v in gauss_top_concepts}
        y_pos_g = np.arange(len(all_names_g))
        ax.barh(y_pos_g - 0.2, [orig_dict_g.get(n, 0) for n in all_names_g], height=0.35, color='steelblue', label='Original', alpha=0.8)
        ax.barh(y_pos_g + 0.2, [gauss_dict.get(n, 0) for n in all_names_g], height=0.35, color='coral', label=f'Isotropic avg', alpha=0.8)
        ax.set_yticks(y_pos_g); ax.set_yticklabels([n[:25] for n in all_names_g], fontsize=9)
        ax.invert_yaxis(); ax.set_xlabel('Activation')
        ax.set_title(f'Isotropic: {get_class_name(PROBE_DATASET, pred_orig)} → '
                     f'{get_class_name(PROBE_DATASET, pred_gauss)} '
                     f'({"STABLE ✓" if pred_orig == pred_gauss else "CHANGED ✗"})',
                     fontsize=12, fontweight='bold')
        ax.set_xlim(0, shared_xlim)
        ax.legend(fontsize=9); ax.grid(True, axis='x', alpha=0.3)
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
            'stable': pred_orig == pred_gauss,
            'mean_overlap': round(float(np.mean(gauss_overlap)), 4),
            'params': {'GAUSS_SIGMA': round(float(gauss_sigma), 4), 'N_SMOOTH_SAMPLES': N_SMOOTH_SAMPLES},
            'original_concepts': [{'name': n, 'value': round(v, 4)} for n, v in orig_concepts],
            'example_noisy_samples': gauss_example_samples,
            'final_avg_concepts': [{'name': n, 'value': round(v, 4)} for n, v in gauss_final_concepts],
        }
        with open(os.path.join(isotropic_dir, f"idx{target_idx}_detail.json"), 'w') as f:
            json.dump(isotropic_result, f, indent=2)

        # Also save input image to both folders
        if img_path and os.path.exists(img_path):
            import shutil
            ext = os.path.splitext(img_path)[1]
            shutil.copy2(img_path, os.path.join(manifold_dir, f"idx{target_idx}_input{ext}"))
            shutil.copy2(img_path, os.path.join(isotropic_dir, f"idx{target_idx}_input{ext}"))

        print(f"\n  Saved viz+detail for idx {target_idx} to: {manifold_dir} & {isotropic_dir}")
        viz_count += 1
    else:
        if (loop_i + 1) % 50 == 0:
            print(f"  [{loop_i+1}/{N_TARGETS}] done")

    results.append({
        'idx': target_idx,
        'label_true': label_true,
        'pred_orig': pred_orig,
        'pred_manifold': pred_smooth,
        'n_votes_manifold': n_votes,
        'stable_manifold': pred_orig == pred_smooth,
        'overlap_manifold': round(float(np.mean(smooth_concepts_overlap)), 4),
        'pred_gaussian': pred_gauss,
        'n_votes_gaussian': n_votes_gauss,
        'stable_gaussian': pred_orig == pred_gauss,
        'overlap_gaussian': round(float(np.mean(gauss_overlap)), 4),
    })

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
n_stable_manifold = sum(r['stable_manifold'] for r in results)
n_stable_gaussian = sum(r['stable_gaussian'] for r in results)
print(f"{'Method':<20s} {'Stable':<12s} {'Mean Overlap':<15s}")
print(f"{'-'*47}")
print(f"{'Manifold':<20s} {str(n_stable_manifold)+'/'+str(len(results)):<12s} "
      f"{np.mean([r['overlap_manifold'] for r in results]):.3f}")
print(f"{'Gaussian (baseline)':<20s} {str(n_stable_gaussian)+'/'+str(len(results)):<12s} "
      f"{np.mean([r['overlap_gaussian'] for r in results]):.3f}")

results_path = os.path.join(SAVE_DIR, f"smoothing_results_{PROBE_DATASET}.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)

# Also save per-method summary
manifold_summary = [{'idx': r['idx'], 'label_true': r['label_true'], 'pred_orig': r['pred_orig'],
                     'pred_smooth': r['pred_manifold'], 'n_votes': r['n_votes_manifold'],
                     'stable': r['stable_manifold'], 'mean_overlap': r['overlap_manifold']} for r in results]
isotropic_summary = [{'idx': r['idx'], 'label_true': r['label_true'], 'pred_orig': r['pred_orig'],
                      'pred_smooth': r['pred_gaussian'], 'n_votes': r['n_votes_gaussian'],
                      'stable': r['stable_gaussian'], 'mean_overlap': r['overlap_gaussian']} for r in results]
with open(os.path.join(manifold_dir, "summary.json"), 'w') as f:
    json.dump(manifold_summary, f, indent=2)
with open(os.path.join(isotropic_dir, "summary.json"), 'w') as f:
    json.dump(isotropic_summary, f, indent=2)

print(f"\nResults saved to:")
print(f"  Combined:  {results_path}")
print(f"  Manifold:  {manifold_dir}/summary.json")
print(f"  Isotropic: {isotropic_dir}/summary.json")
