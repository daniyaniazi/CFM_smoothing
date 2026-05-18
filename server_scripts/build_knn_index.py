"""
Build KNN index on ALL train concept vectors (no subsampling).

Run this ONCE, then run_smoothing.py will load the cached index.

Usage:
  python server_scripts/build_knn_index.py

  # or as SLURM job:
  sbatch server_scripts/build_index_job.sh
"""

import sys
import os
import argparse
import shutil
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# --force flag: rebuild everything even if cached
_parser = argparse.ArgumentParser()
_parser.add_argument('--force', action='store_true', help='Delete existing artifacts and rebuild from scratch')
_cli_args = _parser.parse_args()
FORCE_REBUILD = _cli_args.force

import torch
import numpy as np
import time as _time
import glob
from pathlib import Path
from torch.utils.data import DataLoader

from cfm.arg_parser import get_default_parser
from cfm.utils import common_init, get_img_model, get_probe_dataset
from cfm.cfm import CFM
from cfm import config as cfm_config
from dictionary_learning.utils import load_dictionary

# ===========================================================================
# Config — must match run_smoothing.py
# ===========================================================================
CONFIG_NAME = 'k_12_ef_16_lr_0.0001_mf_[0.008,0.03,0.06,0.12,0.24,0.542]'
PROBE_DATASET = "imagenet"       # or "places365"

BATCH_SIZE = 64
NUM_WORKERS = 4
N_TREES = 20                     # more trees = better accuracy, slower build
CHUNK_SIZE = 5000                # save a chunk every N samples (survive OOM)

SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'smoothing_data')
os.makedirs(SAVE_DIR, exist_ok=True)


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


# ===========================================================================
# Step 2 & 3: Generate concept vectors with CHUNKED SAVING (survives OOM)
# ===========================================================================
# Saves chunks of CHUNK_SIZE to disk as they're produced, then merges at end.
# If interrupted, resumes from the last complete chunk.

args.probe_dataset = PROBE_DATASET
args.probe_dataset_root_dir = cfm_config.probe_dataset_root_dir_dict[PROBE_DATASET]


def generate_concept_vectors_chunked(split_name):
    """Generate concept vectors with intermediate chunk saves + resume."""
    cv_path = os.path.join(SAVE_DIR, f"concept_vectors_{PROBE_DATASET}_{split_name}.pt")
    labels_path = os.path.join(SAVE_DIR, f"labels_{PROBE_DATASET}_{split_name}.pt")
    chunk_dir = os.path.join(SAVE_DIR, f"chunks_{PROBE_DATASET}_{split_name}")

    # Already fully done?
    if FORCE_REBUILD:
        for p in [cv_path, labels_path]:
            if os.path.exists(p):
                os.remove(p)
        if os.path.exists(chunk_dir):
            shutil.rmtree(chunk_dir)
        print(f"\n{split_name}: --force: cleared cached artifacts")
    elif os.path.exists(cv_path) and os.path.exists(labels_path):
        print(f"\n{split_name}: Loading cached vectors from {cv_path}")
        vecs = torch.load(cv_path)
        labs = torch.load(labels_path)
        print(f"  {vecs.shape[0]} vectors, dim={vecs.shape[1]}")
        return vecs, labs

    # Set up chunked generation
    os.makedirs(chunk_dir, exist_ok=True)
    dataset = get_probe_dataset(
        PROBE_DATASET, split_name, args.probe_dataset_root_dir, preprocess_fn=preprocess)
    print(f"\n{split_name}: {len(dataset)} samples")

    # How many samples are already saved in chunks? (resume support)
    existing_chunks = sorted(glob.glob(os.path.join(chunk_dir, "cv_*.pt")))
    n_done = 0
    if existing_chunks:
        for cp in existing_chunks:
            n_done += torch.load(cp).shape[0]
        print(f"  Resuming: found {len(existing_chunks)} chunks ({n_done} vectors already done)")

    loader = DataLoader(dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    chunk_vecs = []
    chunk_labs = []
    chunk_count = len(existing_chunks)
    samples_in_chunk = 0
    samples_seen = 0
    t0 = _time.time()

    with torch.no_grad():
        for batch_idx, (imgs, labs) in enumerate(loader):
            batch_start = samples_seen
            batch_end = samples_seen + imgs.shape[0]
            samples_seen = batch_end

            # Skip already-done samples
            if batch_end <= n_done:
                continue

            imgs = imgs.to(args.device)
            cvs = cfm_model.get_aggregated_concept_activations(imgs)
            chunk_vecs.append(cvs.cpu())
            chunk_labs.append(labs)
            samples_in_chunk += imgs.shape[0]

            del imgs, cvs
            if args.device == 'cuda':
                torch.cuda.empty_cache()

            # Save chunk to disk
            if samples_in_chunk >= CHUNK_SIZE:
                cv_chunk = torch.cat(chunk_vecs, dim=0)
                lb_chunk = torch.cat(chunk_labs, dim=0)
                torch.save(cv_chunk, os.path.join(chunk_dir, f"cv_{chunk_count:05d}.pt"))
                torch.save(lb_chunk, os.path.join(chunk_dir, f"lb_{chunk_count:05d}.pt"))
                total_done = n_done + samples_in_chunk if chunk_count == len(existing_chunks) else batch_end
                elapsed = _time.time() - t0
                print(f"  chunk {chunk_count} saved ({cv_chunk.shape[0]} vecs, "
                      f"~{total_done}/{len(dataset)} total, {elapsed/60:.1f}min)", flush=True)
                chunk_vecs = []
                chunk_labs = []
                samples_in_chunk = 0
                chunk_count += 1

    # Save final partial chunk
    if chunk_vecs:
        cv_chunk = torch.cat(chunk_vecs, dim=0)
        lb_chunk = torch.cat(chunk_labs, dim=0)
        torch.save(cv_chunk, os.path.join(chunk_dir, f"cv_{chunk_count:05d}.pt"))
        torch.save(lb_chunk, os.path.join(chunk_dir, f"lb_{chunk_count:05d}.pt"))
        print(f"  chunk {chunk_count} saved (final, {cv_chunk.shape[0]} vecs)", flush=True)
        del chunk_vecs, chunk_labs

    # Merge all chunks into single files
    print(f"  Merging chunks into {cv_path}...", flush=True)
    all_cv_chunks = sorted(glob.glob(os.path.join(chunk_dir, "cv_*.pt")))
    all_lb_chunks = sorted(glob.glob(os.path.join(chunk_dir, "lb_*.pt")))
    all_vecs = torch.cat([torch.load(p) for p in all_cv_chunks], dim=0)
    all_labs = torch.cat([torch.load(p) for p in all_lb_chunks], dim=0)
    torch.save(all_vecs, cv_path)
    torch.save(all_labs, labels_path)
    print(f"  Saved {all_vecs.shape[0]} {split_name} vectors to {cv_path}")

    # Clean up chunks
    import shutil
    shutil.rmtree(chunk_dir)
    print(f"  Cleaned up chunk dir")

    return all_vecs, all_labs


print("\n" + "=" * 70)
print("Generating concept vectors (chunked + resumable)")
print("=" * 70)

train_concept_vectors, train_labels = generate_concept_vectors_chunked("train")
val_concept_vectors, val_labels = generate_concept_vectors_chunked("val")

# Free the CFM model — not needed for index building
del cfm_model, feature_extractor, autoencoder
if args.device == 'cuda':
    torch.cuda.empty_cache()
print("\nFreed CFM model from memory")


# ===========================================================================
# Step 4: Build Annoy KNN index — STREAMING from chunks or tensor
# ===========================================================================
print("\n" + "=" * 70)
print("Building KNN index on ALL train concept vectors")
print("=" * 70)

import annoy

N_TRAIN = train_concept_vectors.shape[0]
CONCEPT_DIM = train_concept_vectors.shape[1]

index_path = os.path.join(SAVE_DIR, f"knn_concepts_{PROBE_DATASET}_train_{N_TRAIN}.ann")

if os.path.exists(index_path) and not FORCE_REBUILD:
    print(f"Index already exists at {index_path}")
    print("Use --force to rebuild.")
else:
    if os.path.exists(index_path):
        os.remove(index_path)
        print(f"--force: deleted existing index {index_path}")
    knn_index = annoy.AnnoyIndex(CONCEPT_DIM, 'euclidean')

    # Use on_disk_build to write directly to file — halves peak RAM
    knn_index.on_disk_build(index_path)

    # Convert to numpy once, then free the torch tensor
    train_np = train_concept_vectors.numpy()
    del train_concept_vectors, train_labels
    import gc; gc.collect()

    print(f"Adding {N_TRAIN} items (dim={CONCEPT_DIM}) to index...", flush=True)
    t0 = _time.time()
    for i in range(N_TRAIN):
        knn_index.add_item(i, train_np[i])
        if (i + 1) % 200_000 == 0:
            print(f"  added {i+1}/{N_TRAIN} items ({_time.time()-t0:.0f}s)", flush=True)

    del train_np  # free numpy array before the heavy tree-building step
    gc.collect()

    print(f"Building index with {N_TREES} trees (this is the slow part)...", flush=True)
    knn_index.build(N_TREES)
    # No need for knn_index.save() — on_disk_build already writes to index_path
    elapsed = _time.time() - t0
    print(f"\nDone! Built Annoy index in {elapsed/60:.1f} min")
    print(f"  Vectors: {N_TRAIN}")
    print(f"  Dimension: {CONCEPT_DIM}")
    print(f"  Trees: {N_TREES}")
    print(f"  Saved to: {index_path}")
    file_size_mb = Path(index_path).stat().st_size / (1024 * 1024)
    print(f"  File size: {file_size_mb:.1f} MB")

print("\n" + "=" * 70)
print("Index ready. Now update run_smoothing.py to use the full index.")
print("=" * 70)
