"""
Pipeline test: Load CFM model, pass one image, inspect concept vector.
Run on cluster to verify if everything works before running smoothing.

Usage:
  python server_scripts/run_pipeline_test.py
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import numpy as np
from PIL import Image
from pathlib import Path

from cfm.arg_parser import get_default_parser
from cfm.utils import common_init, get_img_model
from cfm.cfm import CFM
from dictionary_learning.utils import load_dictionary

# ===========================================================================
# Config
# ===========================================================================
CONFIG_NAME = 'k_12_ef_16_lr_0.0001_mf_[0.008,0.03,0.06,0.12,0.24,0.542]'

# ===========================================================================
# Step 1: Setup
# ===========================================================================
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

parser = get_default_parser()
args = parser.parse_args([])
args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
common_init(args)
args.config_name = CONFIG_NAME

print(f"Device: {args.device}")
print(f"Image encoder: {args.img_enc_name}")
print(f"SAE checkpoint dir: {args.save_dir_sae_ckpts['img']}")

# ===========================================================================
# Step 2: Load CLIP-DINOiser
# ===========================================================================
feature_extractor, preprocess = get_img_model(args)
feature_extractor.eval()
print("CLIP-DINOiser loaded")

# ===========================================================================
# Step 3: Load SAE
# ===========================================================================
sae_base = Path(args.save_dir_sae_ckpts['img']) / args.config_name / 'trainer_0'
print(f"Looking for SAE at: {sae_base}")
assert sae_base.exists(), f"SAE not found at {sae_base}"
autoencoder, ae_config = load_dictionary(str(sae_base), args.device)
print(f"SAE loaded, config: {ae_config}")

# ===========================================================================
# Step 4: Create CFM model
# ===========================================================================
cfm_model = CFM(
    feature_extractor=feature_extractor,
    autoencoder=autoencoder,
    apply_found=False,
    device=args.device,
)
cfm_model.eval()
print("CFM model ready")

# ===========================================================================
# Step 5: Load concept names (optional)
# ===========================================================================
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
# Step 6: Test on one image
# ===========================================================================
# Use first image from ImageNet val if available, otherwise download a test image
from cfm import config as cfm_config
imagenet_val = os.path.join(cfm_config.probe_dataset_root_dir_dict["imagenet"], "val")
if os.path.exists(imagenet_val):
    img_files = sorted([f for f in os.listdir(imagenet_val) if f.endswith(".JPEG")])[:1]
    if img_files:
        test_img_path = os.path.join(imagenet_val, img_files[0])
        print(f"Using ImageNet val image: {test_img_path}")
    else:
        test_img_path = None
else:
    test_img_path = None

if test_img_path is None:
    import urllib.request
    test_img_path = "/tmp/cfm_test_image.jpg"
    if not os.path.exists(test_img_path):
        url = "https://upload.wikimedia.org/wikipedia/commons/thumb/4/4d/Cat_November_2010-1a.jpg/1200px-Cat_November_2010-1a.jpg"
        urllib.request.urlretrieve(url, test_img_path)
        print(f"Downloaded test image to {test_img_path}")

image = Image.open(test_img_path).convert("RGB")
image_tensor = preprocess(image).unsqueeze(0).to(args.device)
print(f"Image size: {image.size}, tensor shape: {image_tensor.shape}")

# ===========================================================================
# Step 7: Get concept vector
# ===========================================================================
with torch.no_grad():
    concept_vector = cfm_model.get_aggregated_concept_activations(image_tensor)

print("\n" + "=" * 60)
print("CONCEPT VECTOR")
print("=" * 60)
print(f"Shape: {concept_vector.shape}")
print(f"Non-zero: {(concept_vector[0] > 0).sum().item()} / {concept_vector.shape[1]}")
print(f"Max: {concept_vector.max().item():.4f}")
print(f"Mean (non-zero): {concept_vector[concept_vector > 0].mean().item():.4f}")

top_k = 20
top_values, top_indices = concept_vector[0].topk(top_k)
print(f"\nTop {top_k} active concepts:")
for i, (idx, val) in enumerate(zip(top_indices, top_values)):
    name = concept_names[idx.item()] if concept_names else f"concept_{idx.item()}"
    print(f"  {i+1:2d}. [{idx.item():4d}] {name:30s} = {val.item():.4f}")

print("\nPipeline test PASSED")
