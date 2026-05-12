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
TARGET_IDCS = [100, 51, 42, 200, 500]

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
print("Loading dataset and linear classifier (Probe from Kai's folder)")
print("=" * 70)

args.probe_dataset = PROBE_DATASET
args.probe_split = PROBE_SPLIT
args.probe_dataset_root_dir = cfm_config.probe_dataset_root_dir_dict[PROBE_DATASET]

probe_val_dataset = get_probe_dataset(
    PROBE_DATASET, PROBE_SPLIT, args.probe_dataset_root_dir, preprocess_fn=preprocess)
print(f"Dataset: {PROBE_DATASET} ({PROBE_SPLIT}), {len(probe_val_dataset)} samples")

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
# Step 3: Generate concept vectors (GPU, cached)
# ===========================================================================
cv_path = os.path.join(SAVE_DIR, f"concept_vectors_{PROBE_DATASET}_{PROBE_SPLIT}.pt")
labels_path = os.path.join(SAVE_DIR, f"labels_{PROBE_DATASET}_{PROBE_SPLIT}.pt")

if os.path.exists(cv_path) and os.path.exists(labels_path):
    print("\n" + "=" * 70)
    print("Loading cached concept vectors")
    print("=" * 70)
    all_concept_vectors = torch.load(cv_path)
    all_labels = torch.load(labels_path)
    print(f"Loaded {all_concept_vectors.shape[0]} vectors from {cv_path}")
else:
    print("\n" + "=" * 70)
    print("Step 3: Generating concept vectors (this takes a while)")
    print("=" * 70)

    loader = DataLoader(probe_val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    all_concept_vectors = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (imgs, labels) in enumerate(loader):
            imgs = imgs.to(args.device)
            cvs = cfm_model.get_aggregated_concept_activations(imgs)
            all_concept_vectors.append(cvs.cpu())
            all_labels.append(labels)
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch {batch_idx+1}/{len(loader)}")

    all_concept_vectors = torch.cat(all_concept_vectors, dim=0)
    all_labels = torch.cat(all_labels, dim=0)

    torch.save(all_concept_vectors, cv_path)
    torch.save(all_labels, labels_path)
    print(f"Saved {all_concept_vectors.shape[0]} concept vectors to {cv_path}")

print(f"Shape: {all_concept_vectors.shape}")
print(f"Non-zero per vector (mean): {(all_concept_vectors > 0).sum(1).float().mean():.1f}")


# ===========================================================================
# Step 4: Build KNN index (cached)
# ===========================================================================
print("\n" + "=" * 70)
print("Building KNN index")
print("=" * 70)

import annoy

index_path = os.path.join(SAVE_DIR, f"knn_concepts_{PROBE_DATASET}_{PROBE_SPLIT}.ann")
CONCEPT_DIM = all_concept_vectors.shape[1]
N = all_concept_vectors.shape[0]

if os.path.exists(index_path):
    knn_index = annoy.AnnoyIndex(CONCEPT_DIM, 'euclidean')
    knn_index.load(index_path)
    print(f"Loaded existing KNN index from {index_path}")
else:
    knn_index = annoy.AnnoyIndex(CONCEPT_DIM, 'euclidean')
    for i in range(N):
        knn_index.add_item(i, all_concept_vectors[i].numpy())
    N_TREES = 50
    knn_index.build(N_TREES)
    knn_index.save(index_path)
    print(f"Built Annoy index: {N} vectors, dim={CONCEPT_DIM}, trees={N_TREES}")


# ===========================================================================
# Step 5: Manifold smoothing
# ===========================================================================
print("\n" + "=" * 70)
print("Manifold smoothing")
print("=" * 70)
print(f"K={K_NEIGHBORS}, sigma={SCALE_WEIGHT}, N_samples={N_SMOOTH_SAMPLES}")

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

# Save dir per image
per_image_dir = os.path.join(SAVE_DIR, f"per_image_{PROBE_DATASET}")
os.makedirs(per_image_dir, exist_ok=True)

for target_idx in TARGET_IDCS:
    cv_orig = all_concept_vectors[target_idx].numpy()
    label_true = all_labels[target_idx].item()
    pred_orig = classify_concept_vector(
        torch.tensor(cv_orig, device=args.device), classifier_weights)

    # Save input image path (and copy image if possible)
    img_path = get_image_path(probe_val_dataset, target_idx)
    if img_path and os.path.exists(img_path):
        import shutil
        ext = os.path.splitext(img_path)[1]
        dst = os.path.join(per_image_dir, f"idx{target_idx}_input{ext}")
        shutil.copy2(img_path, dst)
        print(f"\n  Saved input image: {dst}")

    # KNN neighbors in concept space
    nn_idcs = knn_index.get_nns_by_item(target_idx, K_NEIGHBORS)
    X_neighbors = np.stack([knn_index.get_item_vector(i) for i in nn_idcs])

    # --- Top 5 neighbors: save their concepts ---
    top5_neighbors_info = []
    for rank, nn_idx in enumerate(nn_idcs[:5]):
        nn_cv = all_concept_vectors[nn_idx].numpy()
        nn_label = all_labels[nn_idx].item()
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

    # --- Print summary ---
    orig_concepts = get_top_concept_info(cv_orig, concept_names, top_k=20)

    print("\n" + "-" * 70)
    print(f"Image idx={target_idx}  |  path: {img_path}")
    print(f"  True class:     {get_class_name(PROBE_DATASET, label_true)}")
    print(f"  Original pred:  {get_class_name(PROBE_DATASET, pred_orig)}")
    print(f"  Smoothed pred:  {get_class_name(PROBE_DATASET, pred_smooth)} "
          f"({n_votes}/{N_SMOOTH_SAMPLES} votes)")
    print(f"  Prediction stable: {pred_orig == pred_smooth}")
    print(f"  Mean top-20 concept overlap: {np.mean(smooth_concepts_overlap):.3f}")

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

    # --- Save detailed per-image JSON ---
    # --- Visualize concept vector manifold ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Project everything to 2D via PCA for visualization
    pca_vis = PCA(n_components=2)
    # Fit on neighbors
    X_vis = pca_vis.fit_transform(X_neighbors)
    # Project original
    orig_vis = pca_vis.transform(cv_orig.reshape(1, -1))[0]
    # Project mean
    mean_vis = pca_vis.transform(mean_nn.reshape(1, -1))[0]

    # Generate noisy samples in original space for visualization
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

    # --- Panel 1: Manifold overview ---
    ax = axes[0]
    ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightblue', s=8, alpha=0.4, label=f'{K_NEIGHBORS} KNN neighbors')
    # Top 5 neighbors
    for rank in range(min(5, len(nn_idcs))):
        nn_cv_vis = pca_vis.transform(all_concept_vectors[nn_idcs[rank]].numpy().reshape(1, -1))[0]
        ax.scatter(nn_cv_vis[0], nn_cv_vis[1], c='blue', s=80, marker='D', zorder=5,
                   edgecolors='darkblue', linewidths=1.5)
        ax.annotate(f'N{rank+1}', (nn_cv_vis[0], nn_cv_vis[1]), fontsize=8, fontweight='bold',
                    xytext=(5, 5), textcoords='offset points')
    ax.scatter(orig_vis[0], orig_vis[1], c='red', s=200, marker='*', zorder=10,
               edgecolors='darkred', linewidths=1.5, label='Target image')
    ax.scatter(mean_vis[0], mean_vis[1], c='green', s=100, marker='X', zorder=10,
               edgecolors='darkgreen', linewidths=1.5, label='Neighborhood mean')
    ax.set_title(f'Concept Vector Manifold (2D PCA)\n'
                 f'Target idx={target_idx}, True: {get_class_name(PROBE_DATASET, label_true)}',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 2: Noisy samples on manifold ---
    ax = axes[1]
    ax.scatter(X_vis[:, 0], X_vis[:, 1], c='lightgray', s=5, alpha=0.2)
    # Color noisy samples by prediction correctness
    colors = ['green' if c else 'orange' for c in noisy_correct]
    ax.scatter(noisy_vis[:, 0], noisy_vis[:, 1], c=colors, s=15, alpha=0.6,
               label=f'{N_SMOOTH_SAMPLES} noisy samples')
    ax.scatter(orig_vis[0], orig_vis[1], c='red', s=200, marker='*', zorder=10,
               edgecolors='darkred', linewidths=1.5, label='Original')
    # Draw spread ellipse
    cov = np.cov(noisy_vis.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    angle = np.degrees(np.arctan2(eigenvectors[1, 1], eigenvectors[0, 1]))
    for n_std in [1, 2]:
        ell = Ellipse(xy=noisy_vis.mean(axis=0), width=2*n_std*np.sqrt(eigenvalues[1]),
                      height=2*n_std*np.sqrt(eigenvalues[0]), angle=angle,
                      fill=False, edgecolor='purple', linestyle='--', linewidth=1.5, alpha=0.6)
        ax.add_patch(ell)
    n_correct = sum(noisy_correct)
    ax.set_title(f'N={N_SMOOTH_SAMPLES} Noisy Samples (σ={SCALE_WEIGHT})\n'
                 f'Green=correct ({n_correct}), Orange=wrong ({N_SMOOTH_SAMPLES-n_correct})',
                 fontsize=12, fontweight='bold')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Panel 3: Concept comparison bar chart ---
    ax = axes[2]
    n_show = 20
    orig_top_concepts = get_top_concept_info(cv_orig, concept_names, top_k=n_show)
    final_top_concepts = get_top_concept_info(cv_smoothed_avg, concept_names, top_k=n_show)

    # Collect all unique concept names from both
    all_names = []
    for n, _ in orig_top_concepts:
        if n not in all_names:
            all_names.append(n)
    for n, _ in final_top_concepts:
        if n not in all_names:
            all_names.append(n)
    all_names = all_names[:30]  # cap

    orig_dict = {n: v for n, v in orig_top_concepts}
    final_dict = {n: v for n, v in final_top_concepts}

    y_pos = np.arange(len(all_names))
    orig_vals = [orig_dict.get(n, 0) for n in all_names]
    final_vals = [final_dict.get(n, 0) for n in all_names]

    ax.barh(y_pos - 0.2, orig_vals, height=0.35, color='steelblue', label='Original', alpha=0.8)
    ax.barh(y_pos + 0.2, final_vals, height=0.35, color='coral', label=f'Smoothed (avg {N_SMOOTH_SAMPLES}x)', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([n[:25] for n in all_names], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Activation')
    ax.set_title(f'Concepts: Original vs Smoothed\n'
                 f'Pred: {get_class_name(PROBE_DATASET, pred_orig)} → '
                 f'{get_class_name(PROBE_DATASET, pred_smooth)} '
                 f'({"STABLE ✓" if pred_orig == pred_smooth else "CHANGED ✗"})',
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, axis='x', alpha=0.3)

    plt.tight_layout()
    fig_path = os.path.join(per_image_dir, f"idx{target_idx}_manifold.png")
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved manifold plot: {fig_path}")

    image_result = {
        'idx': target_idx,
        'image_path': img_path,
        'true_class': get_class_name(PROBE_DATASET, label_true),
        'label_true': label_true,
        'pred_orig': pred_orig,
        'pred_orig_class': get_class_name(PROBE_DATASET, pred_orig),
        'pred_smooth': pred_smooth,
        'pred_smooth_class': get_class_name(PROBE_DATASET, pred_smooth),
        'n_votes': n_votes,
        'stable': pred_orig == pred_smooth,
        'mean_overlap': round(float(np.mean(smooth_concepts_overlap)), 4),
        'params': {
            'K_NEIGHBORS': K_NEIGHBORS,
            'SCALE_WEIGHT': SCALE_WEIGHT,
            'N_SMOOTH_SAMPLES': N_SMOOTH_SAMPLES,
        },
        'original_concepts': [{'name': n, 'value': round(v, 4)} for n, v in orig_concepts],
        'top5_neighbors': top5_neighbors_info,
        'example_noisy_samples': example_noisy_samples,
        'final_avg_concepts': [{'name': n, 'value': round(v, 4)} for n, v in final_concepts],
    }

    per_image_path = os.path.join(per_image_dir, f"idx{target_idx}_detail.json")
    with open(per_image_path, 'w') as f:
        json.dump(image_result, f, indent=2)
    print(f"\n  Saved detail: {per_image_path}")

    results.append({
        'idx': target_idx,
        'label_true': label_true,
        'pred_orig': pred_orig,
        'pred_smooth': pred_smooth,
        'n_votes': n_votes,
        'stable': pred_orig == pred_smooth,
        'mean_overlap': round(float(np.mean(smooth_concepts_overlap)), 4),
    })

# ===========================================================================
# Summary
# ===========================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
n_stable = sum(r['stable'] for r in results)
print(f"Stable predictions: {n_stable}/{len(results)}")
print(f"Mean concept overlap: {np.mean([r['mean_overlap'] for r in results]):.3f}")

results_path = os.path.join(SAVE_DIR, f"smoothing_results_{PROBE_DATASET}.json")
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2)
print(f"Results saved to {results_path}")
