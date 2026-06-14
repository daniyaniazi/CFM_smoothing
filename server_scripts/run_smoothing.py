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
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path
from collections import Counter
from sklearn.decomposition import PCA
from dataclasses import dataclass
from scipy.stats import norm, binomtest, pearsonr, kendalltau
from scipy.stats import beta as beta_dist
from scipy.special import gammaln
import matplotlib
matplotlib.use('Agg')  # no display on server
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

# ── Academic figure style ─────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':      'serif',
    'font.size':        11,
    'axes.titlesize':   12,
    'axes.labelsize':   11,
    'xtick.labelsize':  9,
    'ytick.labelsize':  9,
    'legend.fontsize':  9,
    'figure.facecolor': 'white',
    'axes.facecolor':   'white',
    'axes.spines.top':  False,
    'axes.spines.right':False,
    'axes.grid':        True,
    'grid.alpha':       0.25,
    'grid.linestyle':   '--',
    'lines.linewidth':  2.0,
    'lines.markersize': 5,
    'savefig.dpi':      150,
    'savefig.bbox':     'tight',
})

C = {
    # geometry plots (circle/ellipse)
    'manifold':     '#f5c518',   # yellow/gold  — manifold ellipse
    'isotropic':    '#2166ac',   # deep blue    — iso circle
    'lat_manifold': '#74c476',   # green        — MC samples
    'lat_iso':      '#aec7e8',   # light blue   — iso samples
    'positive':     '#1b7837',
    'negative':     '#762a83',
    'mc':           '#74c476',
    'iso_circle':   '#2166ac',
    'mani_ellipse': '#f5c518',
    'knn':          '#aec7e8',
    'anchor':       'black',
    # concept bars
    'overall':      '#1b7837',   # dark green    — original concept bars
    'match':        '#1b7837',   # dark green    — matched image concepts (same as original)
    # iso: blue shades
    'iso_top':      '#2166ac',   # deep blue     — iso top activated
    'iso_least':    '#d4b483',   # warm sand     — iso least activated (clearly different)
    'iso_bar':      '#2166ac',   # deep blue     — iso noisy query (SAE retrieval)
    # mani: purple top, amber least
    'mani_top':     '#6b3fa0',   # dark purple   — mani top activated
    'mani_least':   '#e08214',   # amber/orange  — mani least activated (clearly different)
    'mani_bar':     '#6b3fa0',   # dark purple   — mani noisy query (SAE retrieval)
    # activation dist boxplots
    'dist_top':     '#6b3fa0',   # dark purple   — top concepts
    'dist_least':   '#e08214',   # amber         — least concepts (warm vs cool = clear contrast)
    'dist_new':     '#2166ac',   # deep blue     — new concepts
}

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
N0_SMOOTH_SAMPLES = int(os.environ.get('CFM_N0_SAMPLES', '50'))   # paper stage-1 class selection
N_SMOOTH_SAMPLES = int(os.environ.get('CFM_N_SAMPLES', '500'))  # supports multi-N via env
N_TARGETS = 500           # number of val images to certify
SAVE_VIZ = os.environ.get('CFM_SAVE_VIZ', '1').strip().lower() not in {'0', 'false', 'no'}
N_VIZ = int(os.environ.get('CFM_N_VIZ', '10'))  # <0 means save viz for all targets
_viz_sigmas_raw = os.environ.get('CFM_VIZ_SIGMAS', '').strip()
if _viz_sigmas_raw:
    try:
        VIZ_SIGMAS = sorted({float(s) for s in _viz_sigmas_raw.replace(';', ',').split(',') if s.strip() and float(s) > 0.0})
    except ValueError:
        print(f"WARNING: Invalid CFM_VIZ_SIGMAS={_viz_sigmas_raw!r}; falling back to current sigma only.")
        VIZ_SIGMAS = [SCALE_WEIGHT]
else:
    VIZ_SIGMAS = [SCALE_WEIGHT]

VIZ_ONLY = os.environ.get('CFM_VIZ_ONLY', '0').strip().lower() in {'1', 'true', 'yes'}

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'smoothing_data')  # shared artifacts (vectors, index)
SAVE_DIR = os.path.join(DATA_DIR, f'sigma_{SCALE_WEIGHT:.2f}_n{N_SMOOTH_SAMPLES}')  # sigma+N specific results
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# ===========================================================================
# Helpers
# ===========================================================================

# --- Paper-aligned certification helpers ---
CERTIFY_ALPHA = 0.001  # confidence level for Clopper-Pearson bounds

def clopper_pearson_lower(n_success, n_total, alpha=CERTIFY_ALPHA):
    """One-sided lower bound on success probability (Clopper-Pearson, Beta)."""
    if n_success == 0:
        return 0.0
    return float(beta_dist.ppf(alpha, n_success, n_total - n_success + 1))

def clopper_pearson_upper(n_success, n_total, alpha=CERTIFY_ALPHA):
    """One-sided upper bound on success probability (Clopper-Pearson, Beta)."""
    if n_success == n_total:
        return 1.0
    return float(beta_dist.ppf(1 - alpha, n_success + 1, n_total - n_success))

def binom_pvalue_two_sided(n_a, n_total, p=0.5):
    """Two-sided binomial test p-value used by the paper PREDICT routine."""
    if n_total <= 0:
        return 1.0
    return float(binomtest(k=int(n_a), n=int(n_total), p=float(p), alternative='two-sided').pvalue)

def predict_from_counts_paper(class_counts, alpha_pred=CERTIFY_ALPHA, abstain_label=-1):
    """Paper PREDICT routine based on top-2 counts and a two-sided binomial test."""
    counts = np.asarray(class_counts, dtype=np.int64)
    if counts.size == 0 or counts.sum() <= 0:
        return abstain_label

    top2 = np.argsort(counts)[-2:]
    c_a = int(top2[-1])
    c_b = int(top2[-2]) if len(top2) > 1 else c_a
    n_a = int(counts[c_a])
    n_b = int(counts[c_b])
    p_val = binom_pvalue_two_sided(n_a, n_a + n_b, p=0.5)
    return c_a if p_val <= float(alpha_pred) else abstain_label

def certified_radius_paper(sigma, p_a_lower):
    """Paper CERTIFY radius: R = σ * Φ⁻¹(p_A_lower), valid when p_A_lower > 0.5."""
    if p_a_lower <= 0.5:
        return 0.0
    return float(sigma) * float(norm.ppf(p_a_lower))

def certify_class_votes_two_stage(class_counts_n0, class_counts_n, sigma,
                                  alpha_conf=CERTIFY_ALPHA,
                                  alpha_pred=CERTIFY_ALPHA,
                                  abstain_label=-1):
    """Paper-exact two-stage CERTIFY routine with extra diagnostics for analysis.

    Stage 1 (n0): choose candidate class from noisy samples.
    Stage 2 (n): estimate a lower confidence bound for that class and certify only if p_A > 0.5.
    """
    counts_n0 = np.asarray(class_counts_n0, dtype=np.int64)
    counts_n = np.asarray(class_counts_n, dtype=np.int64)

    if counts_n0.size == 0 or counts_n.size == 0 or counts_n.sum() <= 0:
        return {
            'top_class': abstain_label,
            'top_class_stage2': abstain_label,
            'n0_votes': 0,
            'n0_votes_runner_up': 0,
            'n_votes': 0,
            'n_votes_runner_up': 0,
            'p_a_lower': 0.0,
            'p_b_upper': 1.0,
            'predict_p_value': 1.0,
            'predict_abstained': True,
            'abstained': True,
            'certified_radius': 0.0,
        }

    top2_n0 = np.argsort(counts_n0)[-2:]
    c_a = int(top2_n0[-1])
    c_b_n0 = int(top2_n0[-2]) if len(top2_n0) > 1 else c_a
    n0_a = int(counts_n0[c_a])
    n0_b = int(counts_n0[c_b_n0])
    predict_p_value = binom_pvalue_two_sided(n0_a, n0_a + n0_b, p=0.5)
    predict_abstained = bool(predict_p_value > float(alpha_pred))

    total_n = int(counts_n.sum())
    n_a = int(counts_n[c_a])
    top2_n = np.argsort(counts_n)[-2:]
    c_top_n = int(top2_n[-1])
    c_b_n = int(top2_n[-2]) if len(top2_n) > 1 else c_top_n
    n_b = int(counts_n[c_b_n])

    p_a_lower = clopper_pearson_lower(n_a, total_n, alpha_conf)
    p_b_upper = 1.0 - p_a_lower
    abstained = bool(not (p_a_lower > 0.5))
    radius = 0.0 if abstained else certified_radius_paper(sigma, p_a_lower)
    pred = abstain_label if abstained else c_a

    return {
        'top_class': pred,
        'candidate_class': c_a,
        'top_class_stage2': c_top_n,
        'n0_votes': n0_a,
        'n0_votes_runner_up': n0_b,
        'n_votes': n_a,
        'n_votes_runner_up': n_b,
        'p_a_lower': round(p_a_lower, 6),
        'p_b_upper': round(p_b_upper, 6),
        'predict_p_value': round(predict_p_value, 6),
        'predict_abstained': predict_abstained,
        'predict_label': predict_from_counts_paper(counts_n0, alpha_pred=alpha_pred, abstain_label=abstain_label),
        'abstained': abstained,
        'certified_radius': round(radius, 6),
    }

def certify_concept(n_survived, n_total, sigma):
    """Certify a single concept's presence in top-K using the paper's p_A-only radius."""
    p_a_lower = clopper_pearson_lower(n_survived, n_total)
    p_b_upper = 1.0 - p_a_lower
    is_certified = bool(p_a_lower > 0.5)
    radius = certified_radius_paper(sigma, p_a_lower)

    return {
        'survival_rate': round(n_survived / n_total, 4),
        'p_a_lower': round(p_a_lower, 6),
        'p_b_upper': round(p_b_upper, 6),
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


# ═══════════════════════════════════════════════════════════════════════════════
# Volume computation  
# ═══════════════════════════════════════════════════════════════════════════════

def log_volume_isotropic(radius: float, k: int) -> float:
    """Log-volume of isotropic certified region (k-ball of radius r).

    V_k(r) = (π^(k/2) / Γ(k/2 + 1)) · r^k
    """
    if radius <= 0.0 or k <= 0:
        return -np.inf
    log_ck = (k / 2.0) * np.log(np.pi) - gammaln(k / 2.0 + 1.0)
    return log_ck + k * np.log(radius)


def log_volume_manifold(radius: float, eigenvalues: np.ndarray) -> float:
    """Log-volume of manifold certified region (ellipsoid).

    V_mani = C_k · r^k · √det(Λ)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    k = len(eigenvalues)
    if radius <= 0.0 or k <= 0:
        return -np.inf
    log_ball = log_volume_isotropic(radius, k)
    log_det_half = 0.5 * np.sum(np.log(np.maximum(eigenvalues, 1e-30)))
    return log_ball + log_det_half


def log_volume_manifold_from_iso_radius(radius_iso: float, eigenvalues: np.ndarray) -> float:
    """Predicted manifold volume with the isotropic radius and raw eigenvalue geometry."""
    return log_volume_manifold(radius_iso, eigenvalues)


def log_volume_ratio(radius_mani: float, radius_iso: float,
                     eigenvalues: np.ndarray) -> float:
    """log(V_mani / V_iso) = k·log(r_mani/r_iso) + 0.5·Σlog(λ_i).

    Two effects:
        - Radius effect:   k·log(r_mani/r_iso)
        - Geometry effect:  0.5·Σlog(λ_i)
    """
    eigenvalues = np.asarray(eigenvalues, dtype=np.float64)
    k = len(eigenvalues)
    if radius_iso <= 0.0 or radius_mani <= 0.0:
        return 0.0
    radius_effect = k * np.log(radius_mani / radius_iso)
    geometry_effect = 0.5 * np.sum(np.log(np.maximum(eigenvalues, 1e-30)))
    return radius_effect + geometry_effect


@dataclass
class EigenDiagnostics:
    """Diagnostic stats for a set of PCA eigenvalues."""
    k: int
    lambda_min: float
    lambda_max: float
    condition_number: float
    eigenvalue_sum: float
    effective_rank: float   # exp(entropy of normalized eigenvalues)
    log_det_half: float     # 0.5 · Σ log(λ_i) — the geometry factor

    def to_dict(self) -> dict:
        return {
            'eigen_k': self.k,
            'eigen_lambda_min': self.lambda_min,
            'eigen_lambda_max': self.lambda_max,
            'eigen_condition_number': self.condition_number,
            'eigen_sum': self.eigenvalue_sum,
            'eigen_effective_rank': self.effective_rank,
            'eigen_log_det_half': self.log_det_half,
        }


def eigenvalue_diagnostics(eigenvalues: np.ndarray) -> EigenDiagnostics:
    """Compute diagnostic statistics for eigenvalues."""
    evals = np.asarray(eigenvalues, dtype=np.float64)
    evals = np.maximum(evals, 1e-30)
    k = len(evals)
    p = evals / evals.sum()
    entropy = -np.sum(p * np.log(p + 1e-30))
    effective_rank = float(np.exp(entropy))
    return EigenDiagnostics(
        k=k,
        lambda_min=float(evals.min()),
        lambda_max=float(evals.max()),
        condition_number=float(evals.max() / evals.min()),
        eigenvalue_sum=float(evals.sum()),
        effective_rank=effective_rank,
        log_det_half=float(0.5 * np.sum(np.log(evals))),
    )


def normalize_eigenvalues(eigenvalues: np.ndarray, mode: str = 'max') -> np.ndarray:
    """Normalize eigenvalues so geometry metrics reflect shape rather than absolute scale."""
    evals = np.asarray(eigenvalues, dtype=np.float64)
    evals = np.maximum(evals, 1e-30)
    if mode == 'max':
        return evals / evals.max()
    if mode == 'mean':
        return evals / evals.mean()
    raise ValueError(f"Unknown normalization mode: {mode!r}. Use 'max' or 'mean'.")


def log_volume_geo_iso(sigma: float, k: int) -> float:
    """Geometry-first isotropic k-ball volume at the same sigma."""
    return log_volume_isotropic(sigma, k)


def log_volume_geo_mani(sigma: float, eigenvalues_norm: np.ndarray) -> float:
    """Geometry-first manifold ellipsoid volume using normalized eigenvalues."""
    evals_norm = np.asarray(eigenvalues_norm, dtype=np.float64)
    k = len(evals_norm)
    if sigma <= 0.0 or k <= 0:
        return -np.inf
    log_det_half_norm = 0.5 * np.sum(np.log(np.maximum(evals_norm, 1e-30)))
    return log_volume_geo_iso(sigma, k) + log_det_half_norm


def axis_lengths(sigma: float, eigenvalues_norm: np.ndarray) -> np.ndarray:
    """Axis lengths a_i = σ * sqrt(λ̃_i) in the local PCA plane."""
    return sigma * np.sqrt(np.maximum(np.asarray(eigenvalues_norm, dtype=np.float64), 0.0))


def anisotropy_ratio(eigenvalues_norm: np.ndarray) -> float:
    """Largest-to-smallest axis ratio of the normalized manifold ellipsoid."""
    evals = np.asarray(eigenvalues_norm, dtype=np.float64)
    evals = np.maximum(evals, 1e-30)
    return float(np.sqrt(evals.max() / evals.min()))


def cumulative_stretch_energy(eigenvalues_norm: np.ndarray, m: int | None = None) -> np.ndarray:
    """Cumulative energy carried by the leading normalized eigenvalues."""
    evals = np.asarray(eigenvalues_norm, dtype=np.float64)
    evals = np.maximum(evals, 0.0)
    total = evals.sum()
    if total <= 0:
        n = len(evals) if m is None else min(m, len(evals))
        return np.zeros(n, dtype=np.float64)
    cumsum = np.cumsum(evals)
    if m is not None:
        cumsum = cumsum[:m]
    return cumsum / total


def compute_volumes(r_iso: float, r_mani: float, eigenvalues: np.ndarray, D: int,
                    sigma: float, normalize_mode: str = 'max') -> dict:
    """Compute certified volume quantities (log-space) per new framework.

    Qty 1: Ambient Iso Ball         = C_D · r_iso^D
    Qty 2: Projected Iso Ball       = C_k · r_iso^k
    Qty 3: Manifold-aware (iso r)   = C_k · r_iso^k · √det(Λ)
    Qty 4: Manifold Ellipsoid       = C_k · r_mani^k · √det(Λ)
    """
    evals = np.asarray(eigenvalues, dtype=np.float64)
    k = len(evals)

    log_qty1 = log_volume_isotropic(r_iso, D)
    log_qty2 = log_volume_isotropic(r_iso, k)
    log_qty3 = log_volume_manifold_from_iso_radius(r_iso, evals)
    log_qty4 = log_volume_manifold(r_mani, evals)

    diag = eigenvalue_diagnostics(evals)
    evals_norm = normalize_eigenvalues(evals, mode=normalize_mode) if k > 0 else np.asarray([], dtype=np.float64)
    geo_iso = log_volume_geo_iso(sigma, k) if k > 0 else -np.inf
    geo_mani = log_volume_geo_mani(sigma, evals_norm) if k > 0 else -np.inf
    axes = axis_lengths(sigma, evals_norm) if k > 0 else np.asarray([], dtype=np.float64)
    cum_energy = cumulative_stretch_energy(evals_norm) if k > 0 else np.asarray([], dtype=np.float64)

    return {
        'log_vol_iso_D': round(float(log_qty1), 4),       # Qty 1
        'log_vol_iso_k': round(float(log_qty2), 4),       # Qty 2
        'log_vol_mani_pred': round(float(log_qty3), 4),   # Qty 3
        'log_vol_mani_actual': round(float(log_qty4), 4), # Qty 4
        'log_vol_ratio': round(float(log_volume_ratio(r_mani, r_iso, evals)), 4),
        'log_vol_geo_iso_k': round(float(geo_iso), 4) if np.isfinite(geo_iso) else float('-inf'),
        'log_vol_geo_mani': round(float(geo_mani), 4) if np.isfinite(geo_mani) else float('-inf'),
        'log_geo_ratio': round(float(geo_mani - geo_iso), 4) if np.isfinite(geo_mani) and np.isfinite(geo_iso) else float('-inf'),
        'r_iso': round(float(r_iso), 6),
        'r_mani': round(float(r_mani), 6),
        'sigma': round(float(sigma), 6),
        'k': k,
        'D': D,
        'normalize_mode': normalize_mode,
        'eigenvalues_norm': [round(float(v), 6) for v in evals_norm.tolist()],
        'axis_lengths': [round(float(v), 6) for v in axes.tolist()],
        'anisotropy_ratio': round(float(anisotropy_ratio(evals_norm)), 6) if k > 0 else 0.0,
        'cumulative_stretch_energy': [round(float(v), 6) for v in cum_energy.tolist()],
        **diag.to_dict(),
    }


def classify_concept_vector(cv, classifier_weights):
    logits = cv @ classifier_weights.T
    return logits.argmax().item()


def get_class_name(probe_dataset, class_idx):
    if class_idx is None or int(class_idx) < 0:
        return "ABSTAIN"
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

# [OLD — 2 Gallery feedback]
# SAE autoencoder was loaded here for decode-then-retrieve viz.
# noisy vectors are OOD for the decoder → decoder output unreliable.
# We cannot tell if bad retrieval is from noise or decoder failure.
# New approach: retrieve DIRECTLY in SAE concept space [8192] — no decoder needed.
#
# try:
#     from dictionary_learning.utils import load_dictionary
#     sae_base = os.path.join(str(args.save_dir_sae_ckpts['img']), args.config_name, 'trainer_0')
#     if os.path.exists(sae_base):
#         autoencoder, _ = load_dictionary(sae_base, args.device)
#         autoencoder.eval()
#         print(f"SAE loaded for decode sanity check from {sae_base}")
#     else:
#         print(f"SAE not found at {sae_base} — decode sanity check will be skipped")
# except Exception as e:
#     print(f"Could not load SAE ({e}) — decode sanity check will be skipped")
#
# # Feature extractor (for true CLIP gallery — panels 4+5 in decode NN figure)
# try:
#     from cfm.utils import get_img_model
#     feature_extractor, _preprocess_fe = get_img_model(args)
#     feature_extractor.eval()
#     print("Feature extractor loaded for true CLIP gallery")
# except Exception as e:
#     print(f"Could not load feature extractor ({e}) — true CLIP gallery will be skipped")
#
# def decode_cv_to_clip(cv_np, ae, device):
#     """cv_np [8192] → clip [512] via SAE linear decoder."""
#     with torch.no_grad():
#         t = torch.tensor(cv_np, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)
#         return ae.decode(t).squeeze().cpu().numpy()

autoencoder = None        # kept as None so VIZ_ONLY code path doesn't crash
feature_extractor = None  # same


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
N_CLASSES = int(classifier_weights.shape[0])

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
# [OLD] Decode-then-retrieve galleries
# decoder is in-distribution trained, noisy vectors are OOD →
#        bad retrieval could be decoder failure, not noise effect.
# ===========================================================================
#
# # Gallery 1: SAE-decoded — val cv [N, 8192] → SAE decode → [N, 512]
# sae_clip_gallery = None
# if autoencoder is not None:
#     with torch.no_grad():
#         cv_t    = val_concept_vectors.to(args.device)
#         decoded = autoencoder.decode(cv_t.unsqueeze(1)).squeeze(1)   # [N, 512]
#         sae_clip_gallery = F.normalize(decoded, dim=-1)
#
# # Gallery 2: true CLIP — val images → feature_extractor → avg pool → [N, 512]
# true_clip_gallery = None
# TRUE_CLIP_CACHE = os.path.join(DATA_DIR, "true_clip_embeddings_imagenet_val.pt")
# if feature_extractor is not None and probe_val_dataset is not None:
#     if os.path.exists(TRUE_CLIP_CACHE):
#         true_clip_gallery = F.normalize(
#             torch.load(TRUE_CLIP_CACHE, map_location=args.device), dim=-1)
#     else:
#         from torch.utils.data import DataLoader as _DL
#         _loader = _DL(probe_val_dataset, batch_size=128, shuffle=False,
#                       num_workers=4, pin_memory=True)
#         _all = []
#         with torch.no_grad():
#             for _imgs, _ in _loader:
#                 _feats = feature_extractor.get_pooled_feats(_imgs.to(args.device))
#                 _all.append(_feats.mean(dim=[2, 3]).cpu())
#         _embs = torch.cat(_all, dim=0)
#         torch.save(_embs, TRUE_CLIP_CACHE)
#         true_clip_gallery = F.normalize(_embs.to(args.device), dim=-1)
#
# def nn_in_gallery(clip_emb_np, gallery):
#     """clip_emb_np [512] → (nn_idx, cosine_sim) in gallery [N, 512]."""
#     with torch.no_grad():
#         q = torch.tensor(clip_emb_np, dtype=torch.float32, device=args.device)
#         q = F.normalize(q, dim=-1)
#         sims = gallery @ q
#         idx = int(sims.argmax().item())
#         return idx, float(sims[idx].item())
#
# def save_decode_nn_figure(target_idx, cv_orig, cv_noisy, sigma, ...):
#     """5-panel figure: original | NN(clean) SAE | NN(noisy) SAE |
#                        NN(clean) TRUE CLIP | NN(noisy) TRUE CLIP"""
#     clean_clip = decode_cv_to_clip(cv_orig,  autoencoder, args.device)
#     noisy_clip = decode_cv_to_clip(cv_noisy, autoencoder, args.device)
#     # ... cosine search in sae_clip_gallery and true_clip_gallery ...


# ===========================================================================
#  SAE concept-space gallery — no decoder, no 512 projection
# ===========================================================================
#
# HOW val concept vectors were built:
#   For every val image:
#     image → CLIP-DINOiser → dense features [196, 512] (14x14 patches)
#           → SAE encoder → sparse activations [196, 8192]
#           → max-pool over 196 patches → [8192]   (one vector per image)
#   Saved to: concept_vectors_imagenet_val.pt  [50000, 8192]
#
# These are already in memory as val_concept_vectors.
#
# WHY L2-normalise for retrieval:
#   cosine_similarity(a, b) = dot(a/|a|, b/|b|)
#   Pre-normalising the gallery means retrieval = one matrix multiply:
#     sims = gallery_norm @ query_norm   [N]   (efficient, no per-pair division)
#   Without normalisation, dot product is dominated by magnitude, not direction.
#   We care about semantic direction (which concepts are active), not raw magnitude.
#
# HOW retrieval gives back an image:
#   argmax(sims) = index i in val set
#   → probe_val_dataset.samples[i][0] = file path on disk
#   → load and display that image
#   The image shown is whatever val image has the closest concept vector to the query.

print("Building SAE concept-space gallery (raw vectors, Euclidean)...", flush=True)
sae_gallery_raw = val_concept_vectors.to(args.device)   # [50000, 8192] — NOT normalised
# Why NOT normalise:
#   - Classifier uses raw dot product — magnitude matters
#   - Annoy KNN index uses Euclidean distance — magnitude matters
#   - L2-normalising hides iso noise effect: iso adds noise to all 8192 dims
#     → huge L2 norm → after normalise looks similar to everything → wrong
#   - Euclidean correctly shows iso vector landed FAR, mani stayed NEAR
print(f"SAE gallery (raw): {sae_gallery_raw.shape}", flush=True)


def top_n_in_sae(cv_np, n=6):
    """
    Retrieve top-n by EUCLIDEAN distance in raw concept space [8192].
    Matches the Annoy KNN index metric used in smoothing certification.
    No normalisation — magnitude differences from iso noise are preserved.
    """
    with torch.no_grad():
        q     = torch.tensor(cv_np, dtype=torch.float32, device=args.device)
        diffs = sae_gallery_raw - q.unsqueeze(0)   # [50000, 8192]
        dists = diffs.norm(dim=1)                   # [50000] Euclidean distances
        top   = dists.topk(n, largest=False)        # smallest distances = nearest
    return top.indices.cpu().tolist(), (-top.values).cpu().tolist()


def save_sae_retrieval_figure(target_idx, cv_orig, cv_iso, cv_mani,
                               sigma, val_labels_t, val_dataset,
                               method_name, save_path, top_n=6, top_k_bars=10,
                               iso_save_dir=None, mani_save_dir=None):
    """
    retrieve directly in SAE concept space [8192].
    No decoder. No 512 projection.

    How it works:
      1. val_concept_vectors [50000, 8192]
             each row = one val image's sparse concept vector
             built by: image → CLIP-DINOiser → SAE encode → max-pool over 196 patches
      2. L2-normalise both query and gallery → cosine similarity = dot product
             Why L2-norm: we care about which concepts are active (direction),
             not how strongly (magnitude). Normalising makes sims comparable.
      3. argmax over [50000] sims → index i → probe_val_dataset.samples[i][0]
             The image file at index i is the one whose concept vector is most
             similar to the query. That is the "retrieved image".

    Figure layout — 4 separate files (iso_top, iso_least, mani_top, mani_least):
      Row 0:   original image | original cv bars  | noisy cv bars  ← fixed reference
      Row 1-5: matched img    | matched TRUE bars  | noisy cv bars  ← top-n SAE matches

      cv_orig is NOT queried — it is in the gallery so top-1 = itself (sim=1.0, trivial).
      cv_iso and cv_mani are queried separately → different matched images → separate figures.
    """
    from PIL import Image as _PILImage
    import matplotlib.gridspec as gridspec

    def _load(idx):
        try:
            if hasattr(val_dataset, 'samples'):
                return _PILImage.open(val_dataset.samples[idx][0]).convert("RGB")
        except Exception:
            pass
        return None

    def _get_bar_vals(cv_np, mode):
        """Return the activation values that will be plotted — used to compute shared xlim."""
        cv = np.array(cv_np)
        if mode == 'top':
            idxs = np.argsort(-cv)[:top_k_bars]
        else:
            active = np.where(cv > 1e-6)[0]
            if len(active) == 0:
                return np.array([0.0])
            idxs = active[np.argsort(cv[active])[:top_k_bars]]
        return cv[idxs]

    def _concept_bars(ax, cv_np, title, color, mode='top', xlim=None):
        cv = np.array(cv_np)
        if mode == 'top':
            idxs = np.argsort(-cv)[:top_k_bars]
        else:
            active = np.where(cv > 1e-6)[0]
            if len(active) == 0:
                ax.set_visible(False)
                return
            idxs = active[np.argsort(cv[active])[:top_k_bars]]
        vals  = cv[idxs]
        names = [(concept_names[i] if concept_names else f"c{i}")[:22] for i in idxs]
        order = np.argsort(vals)[::-1]
        y     = np.arange(len(idxs))
        bars  = ax.barh(y, vals[order], color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([names[i] for i in order], fontsize=6)
        ax.invert_yaxis()
        # shared xlim — same scale across all panels in figure
        xmax = xlim if xlim is not None else (vals.max() or 1) * 1.25
        for bar, val in zip(bars, vals[order]):
            ax.text(xmax * 0.02 + bar.get_width(),
                    bar.get_y() + bar.get_height() / 2,
                    f"{val:.2f}", va='center', ha='left', fontsize=6, color='#333')
        ax.set_xlim(0, xmax)
        ax.set_xlabel('Activation', fontsize=6)
        ax.set_title(title, fontsize=7, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.2, linestyle='--')

    # retrieve top-n only for noisy queries — cv_orig NOT queried (trivial self-match)
    # top_n=6 → 3 rows × 2 cols, no empty space
    iso_idxs,  iso_sims  = top_n_in_sae(cv_iso,  n=top_n)
    mani_idxs, mani_sims = top_n_in_sae(cv_mani, n=top_n)
    label_id   = int(val_labels_t[target_idx].item())
    true_class = get_class_name(PROBE_DATASET, label_id)

    def _single_bars(ax, cv, color, title, mode='top'):
        """
        Clean single-color bar chart — paper style like reference figure.
        One bar per concept. Number at bar end. No overlap. Auto x-axis.
        """
        cv = np.array(cv)
        if mode == 'top':
            idxs = np.argsort(-cv)[:top_k_bars]
        else:
            active = np.where(cv > 1e-6)[0]
            if len(active) == 0:
                ax.text(0.5, 0.5, 'no active concepts', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.set_title(title, fontsize=8, fontweight='bold')
                ax.axis('off')
                return
            idxs = active[np.argsort(cv[active])[:top_k_bars]]

        vals  = cv[idxs]
        names = [(concept_names[i] if concept_names else f"c{i}")[:18] for i in idxs]
        order = np.argsort(vals)[::-1]   # descending — highest at top

        y = np.arange(len(idxs))
        bars = ax.barh(y, vals[order], height=0.6, color=color, alpha=0.85)
        ax.set_yticks(y)
        ax.set_yticklabels([names[i] for i in order], fontsize=7)
        ax.invert_yaxis()

        # x-axis: tight to actual data, not over-extended
        vmax = float(vals.max()) if len(vals) > 0 else 0.01
        xmax = vmax * 1.25 if vmax > 0 else 1.0
        ax.set_xlim(0, xmax)

        # number at end of bar, always visible, no overlap with axis edge
        for bar, val in zip(bars, vals[order]):
            txt = f"{val:.2f}" if val >= 0.01 else f"{val:.4f}"
            ax.text(min(bar.get_width() + vmax * 0.02, xmax * 0.99),
                    bar.get_y() + bar.get_height() / 2,
                    txt, va='center', ha='left', fontsize=6.5,
                    color='#111', clip_on=True)

        ax.set_xlabel('Activation', fontsize=7)
        ax.set_title(title, fontsize=8, fontweight='bold', pad=4)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.2, linestyle='--')

    def _comparison_bars(ax, cv_a, cv_b, color_a, color_b,
                         label_a, label_b, order_by='a', mode='top', xlim=None):
        """Two-bar comparison — used only for original vs noisy panels."""
        cv_a, cv_b = np.array(cv_a), np.array(cv_b)
        ref = cv_a if order_by == 'a' else cv_b
        if mode == 'top':
            idxs = np.argsort(-ref)[:top_k_bars]
        else:
            both_active = np.where((cv_a > 1e-6) | (cv_b > 1e-6))[0]
            if len(both_active) == 0:
                ax.text(0.5, 0.5, 'no active concepts', ha='center', va='center',
                        transform=ax.transAxes, fontsize=8)
                ax.axis('off')
                return
            idxs = both_active[np.argsort(ref[both_active])[:top_k_bars]]
        vals_a = cv_a[idxs]
        vals_b = cv_b[idxs]
        names  = [(concept_names[i] if concept_names else f"c{i}")[:18] for i in idxs]
        order  = np.argsort(ref[idxs])[::-1]
        y, h   = np.arange(len(idxs)), 0.35
        ax.barh(y - h/2, vals_a[order], height=h, color=color_a, alpha=0.85, label=label_a)
        ax.barh(y + h/2, vals_b[order], height=h, color=color_b, alpha=0.85, label=label_b)
        ax.set_yticks(y)
        ax.set_yticklabels([names[i] for i in order], fontsize=7)
        ax.invert_yaxis()
        vmax = max(float(vals_a.max()), float(vals_b.max()), 0.001)
        xmax = xlim if xlim else vmax * 1.25
        for yi, (va, vb) in enumerate(zip(vals_a[order], vals_b[order])):
            for val, yoff in [(va, -h/2), (vb, h/2)]:
                txt = f"{val:.2f}" if val >= 0.01 else f"{val:.4f}"
                ax.text(min(val + vmax * 0.02, xmax * 0.99), yi + yoff,
                        txt, va='center', ha='left', fontsize=6, color='#111', clip_on=True)
        ax.set_xlim(0, xmax)
        ax.set_xlabel('Activation', fontsize=7)
        ax.tick_params(labelsize=7)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.grid(axis='x', alpha=0.2, linestyle='--')

    # Euclidean distances — how far each noisy vector moved from original
    dist_iso  = float(np.linalg.norm(cv_iso  - cv_orig))
    dist_mani = float(np.linalg.norm(cv_mani - cv_orig))

    for mode in ['top', 'least']:
        # mode-aware colours so top and least look different
        c_iso_mode  = C['iso_top']  if mode == 'top' else C['iso_least']
        c_mani_mode = C['mani_top'] if mode == 'top' else C['mani_least']

        for noisy_cv, noisy_idxs, noisy_label, c_noisy, dist, save_dir in [
            (cv_iso,  iso_idxs,  'Iso',      c_iso_mode,  dist_iso,  iso_save_dir  or os.path.dirname(save_path)),
            (cv_mani, mani_idxs, 'Manifold', c_mani_mode, dist_mani, mani_save_dir or os.path.dirname(save_path)),
        ]:
            mode_lbl      = 'Top' if mode == 'top' else 'Least'
            noisy_vec_lbl = f"{noisy_label} noisy vector"
            suptitle = (
                f"{noisy_label} smoothing  {mode_lbl} {top_k_bars} concepts\n"
                f"idx={target_idx}  true: {true_class}  sigma={sigma}\n"
                f"Euclidean dist from original = {dist:.2f}"
            )
            base = f"idx{target_idx}_sae_retrieval_{noisy_label.lower()}_{mode}"

            # ── Original figure: image (row 0) + 2 bar rows (same x-axis) ────
            # Row 0: original image
            # Row 1: ordered by original activation
            # Row 2: ordered by noisy activation
            # Both bar rows share the same xlim so they're directly comparable

            # compute shared xlim across both bar panels
            def _get_vals_for_xlim(cv_a, cv_b, m):
                ref_a = np.array(cv_a)
                ref_b = np.array(cv_b)
                idxs_a = np.argsort(-ref_a)[:top_k_bars] if m == 'top' else \
                         np.where(ref_a > 1e-6)[0][np.argsort(ref_a[np.where(ref_a > 1e-6)[0]])[:top_k_bars]] \
                         if len(np.where(ref_a > 1e-6)[0]) else np.array([], dtype=int)
                idxs_b = np.argsort(-ref_b)[:top_k_bars] if m == 'top' else \
                         np.where(ref_b > 1e-6)[0][np.argsort(ref_b[np.where(ref_b > 1e-6)[0]])[:top_k_bars]] \
                         if len(np.where(ref_b > 1e-6)[0]) else np.array([], dtype=int)
                all_v = []
                if len(idxs_a): all_v += list(ref_a[idxs_a]) + list(np.array(cv_b)[idxs_a])
                if len(idxs_b): all_v += list(ref_b[idxs_b]) + list(np.array(cv_a)[idxs_b])
                return float(max(all_v)) * 1.3 if all_v else 1.0

            shared_xlim0 = _get_vals_for_xlim(cv_orig, noisy_cv, mode)

            fig0, axes0 = plt.subplots(3, 1, figsize=(7, 8),
                                        gridspec_kw={'height_ratios': [0.8, 2, 2],
                                                     'hspace': 0.6})

            # ── Original figure: image top, two single-bar panels below ──────
            # Row 0: original image (small, natural ratio)
            # Row 1: single bars — original cv ordered by original
            # Row 2: single bars — noisy cv ordered by noisy
            fig0, axes0 = plt.subplots(3, 1, figsize=(5, 7),
                                        gridspec_kw={'height_ratios': [0.7, 2, 2],
                                                     'hspace': 0.55})
            ax_img0 = axes0[0]
            img0 = _load(target_idx)
            if img0:
                ax_img0.imshow(img0)
            ax_img0.axis('off')
            ax_img0.set_title(f"ORIGINAL  {true_class[:30]}", fontsize=8, fontweight='bold')

            _single_bars(axes0[1], cv_orig, C['overall'],
                         f"Original concept activations  ({mode_lbl} {top_k_bars})", mode)
            _single_bars(axes0[2], noisy_cv, c_noisy,
                         f"{noisy_label} noisy activations  ({mode_lbl} {top_k_bars})", mode)

            fig0.suptitle(suptitle, fontsize=8, fontweight='bold')
            fig0.savefig(os.path.join(save_dir, f"{base}_original.png"),
                         dpi=150, bbox_inches='tight')
            plt.close(fig0)

            # ── One figure per match — image left, two single-bar panels right ─
            for mi, ni in enumerate(noisy_idxs):
                cv_ni  = val_concept_vectors[ni].numpy()
                lbl_ni = get_class_name(PROBE_DATASET, int(val_labels_t[ni].item()))

                fig_m = plt.figure(figsize=(9, 3.8))
                gs_m  = plt.GridSpec(1, 3, figure=fig_m,
                                     width_ratios=[1, 2, 2],
                                     wspace=0.35)

                ax_img_m = fig_m.add_subplot(gs_m[0])
                img_m = _load(ni)
                if img_m: ax_img_m.imshow(img_m)
                ax_img_m.axis('off')
                ax_img_m.set_title(f"Match #{mi+1}\n{lbl_ni[:20]}", fontsize=8, fontweight='bold')

                # matched image TRUE bars
                _single_bars(fig_m.add_subplot(gs_m[1]), cv_ni, C['overall'],
                             f"Matched #{mi+1} true concepts  ({mode_lbl})", mode)
                # noisy query bars
                _single_bars(fig_m.add_subplot(gs_m[2]), noisy_cv, c_noisy,
                             f"{noisy_label} noisy activations  ({mode_lbl})", mode)

                fig_m.suptitle(suptitle, fontsize=8, fontweight='bold')
                fig_m.savefig(os.path.join(save_dir, f"{base}_match{mi+1}.png"),
                              dpi=150, bbox_inches='tight')
                plt.close(fig_m)


def save_match_grid_figure(target_idx, cv_orig, cv_iso, cv_mani,
                            sigma, val_labels_t, val_dataset, save_dir,
                            top_n=6):
    """
    Combined image grid — paper-style.
    Layout: 3 rows × (top_n+1) cols

      Row 0: original | manifold match 1 | match 2 | ... | match N
      Row 1: (blank)  | iso      match 1 | match 2 | ... | match N

    Saved as: idx{N}_match_grid.png  in save_dir
    """
    from PIL import Image as _PILImage

    def _load(idx):
        try:
            if hasattr(val_dataset, 'samples'):
                return _PILImage.open(val_dataset.samples[idx][0]).convert("RGB")
        except Exception:
            pass
        return None

    # retrieve
    mani_idxs, _ = top_n_in_sae(cv_mani, n=top_n)
    iso_idxs,  _ = top_n_in_sae(cv_iso,  n=top_n)

    label_id   = int(val_labels_t[target_idx].item())
    true_class = get_class_name(PROBE_DATASET, label_id)

    # Layout:
    #   Row 0: original image (centred, spans full width)
    #   Row 1: manifold match 1 … match N
    #   Row 2: isotropic match 1 … match N

    fig = plt.figure(figsize=(2.2 * top_n, 7.5))
    gs  = plt.GridSpec(3, top_n, figure=fig,
                       height_ratios=[1.2, 1, 1],
                       hspace=0.45, wspace=0.05)

    # row 0: original image — first column only
    ax_orig = fig.add_subplot(gs[0, 0])
    img0 = _load(target_idx)
    if img0: ax_orig.imshow(img0)
    ax_orig.axis('off')
    ax_orig.set_title(f"ORIGINAL\n{true_class[:22]}", fontsize=7, fontweight='bold')

    # row 1: manifold matches
    for mi, ni in enumerate(mani_idxs):
        ax = fig.add_subplot(gs[1, mi])
        img = _load(ni)
        if img: ax.imshow(img)
        ax.axis('off')
        lbl = get_class_name(PROBE_DATASET, int(val_labels_t[ni].item()))
        ax.set_title(f"Mani #{mi+1}\n{lbl[:16]}", fontsize=5.5)
        if mi == 0:
            ax.set_ylabel('Manifold', fontsize=6, fontweight='bold', labelpad=4)

    # row 2: iso matches
    for mi, ni in enumerate(iso_idxs):
        ax = fig.add_subplot(gs[2, mi])
        img = _load(ni)
        if img: ax.imshow(img)
        ax.axis('off')
        lbl = get_class_name(PROBE_DATASET, int(val_labels_t[ni].item()))
        ax.set_title(f"Iso #{mi+1}\n{lbl[:16]}", fontsize=5.5)
        if mi == 0:
            ax.set_ylabel('Isotropic', fontsize=6, fontweight='bold', labelpad=4)

    fig.suptitle(
        f"SAE Space Match  idx={target_idx}  true: {true_class}\n"
        f"sigma={sigma}   dist_mani={float(np.linalg.norm(cv_mani - cv_orig)):.2f}"
        f"   dist_iso={float(np.linalg.norm(cv_iso - cv_orig)):.2f}",
        fontsize=8, fontweight='bold'
    )
    fig.savefig(os.path.join(save_dir, f"idx{target_idx}_match_grid.png"),
                dpi=150, bbox_inches='tight')
    plt.close(fig)


# ===========================================================================
# Step 5: Manifold smoothing
# ===========================================================================
print("\n" + "=" * 70, flush=True)
print("Manifold smoothing", flush=True)
print("=" * 70, flush=True)
print(f"K={K_NEIGHBORS}, sigma={SCALE_WEIGHT}, n0={N0_SMOOTH_SAMPLES}, n={N_SMOOTH_SAMPLES}", flush=True)
print(f"Index: TRAIN ({N_TRAIN} vectors), Targets: VAL", flush=True)
_viz_limit_label = 'all' if N_VIZ < 0 else str(N_VIZ)
print(f"Visualization: enabled={SAVE_VIZ}, limit={_viz_limit_label}, sigmas={VIZ_SIGMAS}", flush=True)

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
if SAVE_VIZ:
    _viz_text = 'all targets' if N_VIZ < 0 else f'first {N_VIZ}'
else:
    _viz_text = 'disabled'
print(f"Certifying {N_TARGETS} val images (viz: {_viz_text})", flush=True)
viz_count = 0

# ===========================================================================
# VIZ-ONLY mode: reload existing JSONL results and regenerate all figures
# without rerunning any smoothing computation.
# ===========================================================================
if VIZ_ONLY:
    print("\n" + "=" * 70, flush=True)
    print("VIZ-ONLY mode: regenerating figures from existing JSONL", flush=True)
    print("=" * 70, flush=True)

    _target_set = set(TARGET_IDCS)
    _viz_only_results = []
    if os.path.exists(results_jsonl_path):
        with open(results_jsonl_path, 'r') as _f:
            for _line in _f:
                _line = _line.strip()
                if _line:
                    try:
                        _r = json.loads(_line)
                        if _r['idx'] in _target_set:
                            _viz_only_results.append(_r)
                    except json.JSONDecodeError:
                        pass
    print(f"Loaded {len(_viz_only_results)} results to regenerate viz for", flush=True)

    for _viz_r in _viz_only_results:
        target_idx  = _viz_r['idx']
        label_true  = _viz_r['label_true']
        pred_orig   = _viz_r['pred_orig']
        pred_smooth = _viz_r.get('pred_manifold', _viz_r.get('top_class', -1))
        pred_gauss  = _viz_r['pred_gaussian']
        cv_orig     = val_concept_vectors[target_idx].numpy()

        save_viz = SAVE_VIZ and (N_VIZ < 0 or viz_count < N_VIZ)
        if not save_viz:
            break

        # Recompute KNN/PCA manifold geometry
        nn_idcs     = knn_index.get_nns_by_vector(cv_orig.tolist(), K_NEIGHBORS)
        X_neighbors = np.stack([train_concept_vectors[i].numpy() for i in nn_idcs])
        mean_nn     = X_neighbors.mean(axis=0)
        X_centered  = X_neighbors - mean_nn
        pca         = PCA(n_components=K_NEIGHBORS)
        pca.fit(X_centered)
        ev  = pca.explained_variance_
        Vt  = pca.components_

        cv_whitened  = (cv_orig - mean_nn) @ Vt.T / np.sqrt(ev)
        lambda_max      = float(ev[0])
        sqrt_lambda_max = float(np.sqrt(max(lambda_max, 1e-12)))
        alpha           = SCALE_WEIGHT / sqrt_lambda_max

        # PCA projection for 2-D overlay
        pca_vis  = PCA(n_components=2)
        X_vis    = pca_vis.fit_transform(X_neighbors)
        orig_vis = pca_vis.transform(cv_orig.reshape(1, -1))[0]

        ev_vis_norm = np.asarray(pca_vis.explained_variance_[:2], dtype=np.float64)
        if ev_vis_norm.size < 2:
            ev_vis_norm = np.asarray([1.0, 1.0], dtype=np.float64)
        ev_vis_norm = np.maximum(ev_vis_norm, 1e-12)
        ev_vis_norm = ev_vis_norm / ev_vis_norm.max()

        ev_full_norm = np.maximum(ev, 1e-30) / float(ev[0]) if ev.size > 0 else np.array([1.0])
        gauss_sigma  = SCALE_WEIGHT

        # Recompute averaged smoothed vectors
        cv_smoothed_accum = np.zeros_like(cv_orig)
        for _ in range(N_SMOOTH_SAMPLES):
            _noise = np.random.normal(0, alpha, size=len(ev))
            _cv_n  = cv_whitened + _noise
            cv_smoothed_accum += _cv_n @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
        cv_smoothed_avg = cv_smoothed_accum / N_SMOOTH_SAMPLES

        cv_gauss_accum = np.zeros_like(cv_orig)
        for _ in range(N_SMOOTH_SAMPLES):
            _noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
            cv_gauss_accum += cv_orig + _noise
        cv_gauss_avg = cv_gauss_accum / N_SMOOTH_SAMPLES

        # Recompute noisy sample clouds for overlay figures
        vis_noisy_points = []
        vis_noisy_preds  = []
        for _ in range(N_SMOOTH_SAMPLES):
            _noise = np.random.normal(0, alpha, size=len(ev))
            _cv_s  = (cv_whitened + _noise) @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
            vis_noisy_points.append(_cv_s)
            vis_noisy_preds.append(classify_concept_vector(
                torch.tensor(_cv_s, dtype=torch.float32, device=args.device),
                classifier_weights))
        noisy_vis = pca_vis.transform(np.stack(vis_noisy_points))

        vis_gauss_points     = []
        vis_gauss_preds_list = []
        for _ in range(N_SMOOTH_SAMPLES):
            _noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
            _cv_g  = cv_orig + _noise
            vis_gauss_points.append(_cv_g)
            vis_gauss_preds_list.append(classify_concept_vector(
                torch.tensor(_cv_g, dtype=torch.float32, device=args.device),
                classifier_weights))
        gauss_vis = pca_vis.transform(np.stack(vis_gauss_points))

        print(f"  Regenerating viz for idx={target_idx}", flush=True)

        # ── zoom helpers (duplicated here so they're available outside the main loop) ──
        def _zoom_setup_vo(neighbors_2d, sigmas):
            pc1_std   = float(np.std(neighbors_2d[:, 0]))
            pc2_std   = float(np.std(neighbors_2d[:, 1]))
            pc1_range = float(np.max(neighbors_2d[:, 0]) - np.min(neighbors_2d[:, 0]))
            max_sigma = float(max(sigmas)) if sigmas else float(SCALE_WEIGHT)
            zoom_mid  = 0.10 * pc1_range if pc1_range > 0 else max_sigma
            zoom_tight = max_sigma
            return pc1_std, pc2_std, max_sigma, zoom_mid, zoom_tight

        def _apply_zoom_limits_vo(ax, zr, anchor_2d, neighbors_2d):
            if zr is None:
                x_all = np.concatenate([neighbors_2d[:, 0], [anchor_2d[0]]])
                y_all = np.concatenate([neighbors_2d[:, 1], [anchor_2d[1]]])
                pad   = 0.05 * max(float(x_all.max()-x_all.min()), float(y_all.max()-y_all.min()), 1e-6)
                ax.set_xlim(float(x_all.min())-pad, float(x_all.max())+pad)
                ax.set_ylim(float(y_all.min())-pad, float(y_all.max())+pad)
            else:
                ax.set_xlim(float(anchor_2d[0])-zr, float(anchor_2d[0])+zr)
                ax.set_ylim(float(anchor_2d[1])-zr, float(anchor_2d[1])+zr)

        def _save_iso_overlay_vo(neighbors_2d, anchor_2d, sample_points_2d, sigmas, save_path):
            neighbors_2d     = np.asarray(neighbors_2d,     dtype=np.float64)
            anchor_2d        = np.asarray(anchor_2d,        dtype=np.float64)
            sample_points_2d = np.asarray(sample_points_2d, dtype=np.float64)
            if neighbors_2d.shape[0] == 0:
                return
            pc1_std, pc2_std, max_sigma, zoom_mid, zoom_tight = _zoom_setup_vo(neighbors_2d, sigmas)
            sigma_colors = [C['iso_circle']] * max(len(sigmas), 1)
            zoom_configs = [
                (None,       f"(a) Full noise cloud   σ/std={max_sigma/(pc1_std+1e-12):.2f}"),
                (zoom_mid,   f"(b) Mid-zoom  ±{zoom_mid:.3f}"),
                (zoom_tight, f"(c) Tight zoom  ±σ={zoom_tight:.3f}"),
            ]
            fig_i, axs_i = plt.subplots(1, 3, figsize=(16, 5.5), facecolor='white')
            for ax, (zr, title) in zip(axs_i, zoom_configs):
                ax.scatter(neighbors_2d[:, 0], neighbors_2d[:, 1], c=C['knn'], s=6,
                           alpha=0.28, linewidths=0, zorder=1, label='KNN neighbors')
                if sample_points_2d.size > 0:
                    ax.scatter(sample_points_2d[:, 0], sample_points_2d[:, 1],
                               c=C['lat_iso'], s=9, alpha=0.35, linewidths=0, zorder=2,
                               label='Iso noisy samples')
                for si, sigma in enumerate(sigmas):
                    ax.add_patch(plt.Circle(
                        (float(anchor_2d[0]), float(anchor_2d[1])), float(sigma),
                        fill=False, edgecolor=sigma_colors[si], linewidth=2.0,
                        linestyle=(0, (4, 2)), alpha=0.95, zorder=4,
                        label=f'Iso circle r=σ={sigma:.3f}' if si == 0 else None,
                    ))
                ax.scatter(float(anchor_2d[0]), float(anchor_2d[1]), c=C['anchor'], s=170,
                           marker='*', edgecolors='#444', linewidths=0.8, zorder=6, label='Anchor')
                _apply_zoom_limits_vo(ax, zr, anchor_2d, neighbors_2d)
                ax.set_aspect('equal')
                ax.set_title(title, fontsize=8.5)
                ax.set_xlabel(f'PC1 (cloud std={pc1_std:.2f})')
                ax.set_ylabel(f'PC2 (cloud std={pc2_std:.2f})')
                ax.grid(alpha=0.3)
                ax.legend_.remove() if ax.legend_ else None
            # legend at bottom — shared across all panels
            legend_handles_i = [
                plt.Line2D([0],[0], color=C['lat_iso'], marker='o', lw=0, markersize=5,
                           label=f'Iso MC samples (n={N_SMOOTH_SAMPLES})'),
                plt.Line2D([0],[0], color=C['iso_circle'], lw=2.0, linestyle=(0,(4,2)),
                           label=f'σ-circle r=σ'),
                plt.Line2D([0],[0], marker='*', color='w', markerfacecolor=C['anchor'],
                           markeredgecolor='#444', markersize=10, label='Anchor'),
            ]
            fig_i.legend(handles=legend_handles_i, loc='lower center', ncol=3, fontsize=8,
                         frameon=True, framealpha=0.95, edgecolor='#cccccc',
                         bbox_to_anchor=(0.5, -0.03))
            fig_i.suptitle(
                f"Isotropic Smoothing Geometry   σ={SCALE_WEIGHT}   MC={N_SMOOTH_SAMPLES}",
                fontsize=11, fontweight='bold', y=1.03,
            )
            plt.tight_layout(rect=[0, 0.06, 1, 1])
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close(fig_i)

        def _save_manifold_overlay_vo(neighbors_2d, anchor_2d, sample_points_2d,
                                      evals_norm_2d, sigmas, save_path, evals_full_norm=None):
            neighbors_2d     = np.asarray(neighbors_2d,     dtype=np.float64)
            anchor_2d        = np.asarray(anchor_2d,        dtype=np.float64)
            sample_points_2d = np.asarray(sample_points_2d, dtype=np.float64)
            if neighbors_2d.shape[0] == 0:
                return
            pc1_std, pc2_std, max_sigma, zoom_mid, zoom_tight = _zoom_setup_vo(neighbors_2d, sigmas)
            if evals_full_norm is not None and len(evals_full_norm) > 0:
                mid_idx      = len(evals_full_norm) // 2
                ev_mid_norm  = float(evals_full_norm[mid_idx])
                ev_last_norm = float(evals_full_norm[-1])
                n_last       = len(evals_full_norm) - 1
            else:
                mid_idx      = 0
                ev_mid_norm  = float(evals_norm_2d[-1]) if len(evals_norm_2d) > 1 else 1.0
                ev_last_norm = ev_mid_norm
                n_last       = '?'
            a_mid  = max_sigma * float(np.sqrt(max(ev_mid_norm,  0.0)))
            a_last = max_sigma * float(np.sqrt(max(ev_last_norm, 0.0)))
            zoom_configs = [
                (None,       f"(a) Full neighbourhood cloud"),
                (zoom_mid,   f"(b) Mid-zoom  ±{zoom_mid:.2f}"),
                (zoom_tight, f"(c) Ellipse axes  a₁={max_sigma:.3f}  a₂={max_sigma*float(np.sqrt(max(evals_norm_2d[1],0))) if len(evals_norm_2d)>1 else 0:.3f}"),
                (zoom_tight, f"(d) Mid axis  a_mid={a_mid:.3f}  a_mid/a₁={a_mid/(max_sigma+1e-12):.3f}"),
                (zoom_tight, f"(e) Last axis  a_last={a_last:.2e}  a_last/a₁={a_last/(max_sigma+1e-12):.2e}"),
            ]
            _n_sigmas    = max(len(sigmas), 1)
            sigma_alphas = np.linspace(0.5, 0.95, _n_sigmas)
            fig_m, axs_m = plt.subplots(1, 5, figsize=(27, 5.5), facecolor='white')
            for panel_idx, (ax, (zr, panel_title)) in enumerate(zip(axs_m, zoom_configs)):
                ax.scatter(neighbors_2d[:, 0], neighbors_2d[:, 1], c=C['knn'], s=6,
                           alpha=0.28, linewidths=0, zorder=1, label='KNN neighbors')
                if sample_points_2d.size > 0:
                    ax.scatter(sample_points_2d[:, 0], sample_points_2d[:, 1],
                               c=C['mc'], s=9, alpha=0.35, linewidths=0, zorder=2,
                               label='Manifold noisy samples')
                for si, sigma in enumerate(sigmas):
                    axes_2d = axis_lengths(sigma, evals_norm_2d)
                    ax.add_patch(plt.Circle(
                        (float(anchor_2d[0]), float(anchor_2d[1])), float(sigma),
                        fill=False, edgecolor=C['iso_circle'], linewidth=1.8,
                        linestyle=(0, (4, 2)), alpha=float(sigma_alphas[si]), zorder=4,
                        label=f'Iso circle r=σ={sigma:.3f}' if si == 0 else None,
                    ))
                    if len(axes_2d) >= 2:
                        if panel_idx == 3:
                            ell_h   = float(2 * a_mid)
                            ell_lbl = f'Ellipse a1={axes_2d[0]:.4f} a_mid={a_mid:.4f}'
                        elif panel_idx == 4:
                            ell_h   = float(2 * a_last)
                            ell_lbl = f'Ellipse a1={axes_2d[0]:.4f} a_last={a_last:.4f}'
                        else:
                            ell_h   = float(2 * axes_2d[1])
                            ell_lbl = f'Ellipse a1={axes_2d[0]:.4f} a2={axes_2d[1]:.4f}'
                        ax.add_patch(Ellipse(
                            (float(anchor_2d[0]), float(anchor_2d[1])),
                            width=float(2 * axes_2d[0]), height=ell_h,
                            fill=False, edgecolor=C['mani_ellipse'], linewidth=1.8,
                            linestyle='solid', alpha=float(sigma_alphas[si]), zorder=4,
                            label=ell_lbl if si == 0 else None,
                        ))
                ax.scatter(float(anchor_2d[0]), float(anchor_2d[1]), c=C['anchor'], s=170,
                           marker='*', edgecolors='#444', linewidths=0.8, zorder=6, label='Anchor')
                _apply_zoom_limits_vo(ax, zr, anchor_2d, neighbors_2d)
                ax.set_aspect('equal')
                ax.set_title(panel_title, fontsize=8.5)
                ax.set_xlabel(f'PC1 (cloud std={pc1_std:.2f})')
                ax.set_ylabel(f'PC2 (cloud std={pc2_std:.2f})')
                ax.grid(alpha=0.3)
                # no per-panel legend — shared legend at bottom of figure
            legend_handles_m = [
                plt.Line2D([0], [0], color=C['iso_circle'], lw=1.8, linestyle=(0, (4, 2)), label='Iso circle r=σ'),
                plt.Line2D([0], [0], color=C['mani_ellipse'], lw=1.8, linestyle='solid', label='Manifold ellipse'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C['mc'],
                           markersize=7, label='Manifold noisy samples'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=C['anchor'],
                           markeredgecolor='#444', markersize=10, label='Anchor'),
            ]
            fig_m.legend(handles=legend_handles_m, loc='lower center', ncol=4, fontsize=8,
                         frameon=True, framealpha=0.95, edgecolor='#cccccc', bbox_to_anchor=(0.5, -0.03))
            fig_m.suptitle(
                f"Manifold Smoothing Geometry   σ={SCALE_WEIGHT}\n"
                f"λ_max={lambda_max:.4f}   √λ_max={sqrt_lambda_max:.4f}   α={alpha:.4f}",
                fontsize=11, fontweight='bold', y=1.03,
            )
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close(fig_m)

        # ── geometry overlays ─────────────────────────────────────────────
        _save_manifold_overlay_vo(
            X_vis, orig_vis, noisy_vis,
            ev_vis_norm, VIZ_SIGMAS,
            os.path.join(manifold_dir, f"idx{target_idx}_circle_ellipse.png"),
            evals_full_norm=ev_full_norm,
        )
        _save_iso_overlay_vo(
            X_vis, orig_vis, gauss_vis, VIZ_SIGMAS,
            os.path.join(isotropic_dir, f"idx{target_idx}_circle_ellipse.png"),
        )

        # ── manifold bar chart ────────────────────────────────────────────
        n_show = 20
        smooth_top_idxs = np.argsort(-cv_smoothed_avg)[:n_show]
        top_names_vo    = [(concept_names[i] if concept_names else f"c_{i}") for i in smooth_top_idxs]
        top_orig_vals   = [float(cv_orig[i])         for i in smooth_top_idxs]
        top_smooth_vals = [float(cv_smoothed_avg[i]) for i in smooth_top_idxs]

        active_mask_m  = cv_orig > 1e-6
        active_idxs_m  = np.where(active_mask_m)[0]
        least_idxs_m   = sorted(active_idxs_m, key=lambda i: cv_smoothed_avg[i])[:n_show]
        least_names_m  = [(concept_names[i] if concept_names else f"c_{i}") for i in least_idxs_m]
        least_orig_m   = [float(cv_orig[i])         for i in least_idxs_m]
        least_smooth_m = [float(cv_smoothed_avg[i]) for i in least_idxs_m]

        shared_xlim_m = max(top_orig_vals + top_smooth_vals + least_orig_m + least_smooth_m) * 1.08
        fig_bm, axes_bm = plt.subplots(1, 2, figsize=(18, 8))
        y = np.arange(n_show)

        ax = axes_bm[0]
        ax.barh(y - 0.2, top_orig_vals[::-1],   height=0.35, color=C['overall'],   label='Original',     alpha=0.8)
        ax.barh(y + 0.2, top_smooth_vals[::-1],  height=0.35, color=C['mani_top'],  label='Manifold avg', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels([n[:25] for n in top_names_vo[::-1]], fontsize=8)
        ax.set_xlim(0, shared_xlim_m); ax.set_xlabel('Activation')
        ax.set_title(f'Top Activated  [{get_class_name(PROBE_DATASET, pred_orig)} → '
                     f'{get_class_name(PROBE_DATASET, pred_smooth)} '
                     f'{"STABLE ✓" if pred_orig == pred_smooth else "CHANGED ✗"}]',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        ax = axes_bm[1]
        ax.barh(y - 0.2, least_orig_m[::-1],   height=0.35, color=C['overall'],    label='Original',     alpha=0.8)
        ax.barh(y + 0.2, least_smooth_m[::-1],  height=0.35, color=C['mani_least'], label='Manifold avg', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels([n[:25] for n in least_names_m[::-1]], fontsize=8)
        ax.set_xlim(0, shared_xlim_m); ax.set_xlabel('Activation')
        ax.set_title('Least Activated (originally active, sorted by smoothed score)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        fig_bm.suptitle(
            f'Manifold Smoothing   σ={SCALE_WEIGHT}\n'
            f'True: {get_class_name(PROBE_DATASET, label_true)}   '
            f'Predicted: {get_class_name(PROBE_DATASET, pred_smooth)}',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(manifold_dir, f"idx{target_idx}_top_least_activation.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig_bm)

        # ── isotropic bar chart ───────────────────────────────────────────
        gauss_top_idxs  = np.argsort(-cv_gauss_avg)[:n_show]
        g_top_names_vo  = [(concept_names[i] if concept_names else f"c_{i}") for i in gauss_top_idxs]
        g_top_orig_vals   = [float(cv_orig[i])      for i in gauss_top_idxs]
        g_top_smooth_vals = [float(cv_gauss_avg[i]) for i in gauss_top_idxs]

        active_mask_g  = cv_orig > 1e-6
        active_idxs_g  = np.where(active_mask_g)[0]
        least_idxs_g   = sorted(active_idxs_g, key=lambda i: cv_gauss_avg[i])[:n_show]
        least_names_g  = [(concept_names[i] if concept_names else f"c_{i}") for i in least_idxs_g]
        least_orig_g   = [float(cv_orig[i])      for i in least_idxs_g]
        least_smooth_g = [float(cv_gauss_avg[i]) for i in least_idxs_g]

        shared_xlim_g = max(g_top_orig_vals + g_top_smooth_vals + least_orig_g + least_smooth_g) * 1.08
        fig_bg, axes_bg = plt.subplots(1, 2, figsize=(18, 8))
        y = np.arange(n_show)

        ax = axes_bg[0]
        ax.barh(y - 0.2, g_top_orig_vals[::-1],   height=0.35, color=C['overall'],  label='Original',      alpha=0.8)
        ax.barh(y + 0.2, g_top_smooth_vals[::-1],  height=0.35, color=C['iso_top'],  label='Isotropic avg', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels([n[:25] for n in g_top_names_vo[::-1]], fontsize=8)
        ax.set_xlim(0, shared_xlim_g); ax.set_xlabel('Activation')
        ax.set_title(f'Top Activated  [{get_class_name(PROBE_DATASET, pred_orig)} → '
                     f'{get_class_name(PROBE_DATASET, pred_gauss)} '
                     f'{"STABLE ✓" if pred_orig == pred_gauss else "CHANGED ✗"}]',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        ax = axes_bg[1]
        ax.barh(y - 0.2, least_orig_g[::-1],   height=0.35, color=C['overall'],   label='Original',      alpha=0.8)
        ax.barh(y + 0.2, least_smooth_g[::-1],  height=0.35, color=C['iso_least'], label='Isotropic avg', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels([n[:25] for n in least_names_g[::-1]], fontsize=8)
        ax.set_xlim(0, shared_xlim_g); ax.set_xlabel('Activation')
        ax.set_title('Least Activated (originally active, sorted by smoothed score)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        fig_bg.suptitle(
            f'Manifold Smoothing   σ={SCALE_WEIGHT}\n'
            f'True: {get_class_name(PROBE_DATASET, label_true)}   '
            f'Predicted: {get_class_name(PROBE_DATASET, pred_gauss)}',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(isotropic_dir, f"idx{target_idx}_top_least_activation.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig_bg)

        # ── SAE-space retrieval + match grid figures ─────────────────────────
        if probe_val_dataset is not None:
            cv_noisy_m = (cv_whitened + np.random.normal(0, alpha, size=len(ev))) @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
            cv_noisy_g = cv_orig + np.random.normal(0, gauss_sigma, size=cv_orig.shape)
            save_sae_retrieval_figure(
                target_idx, cv_orig, cv_noisy_g, cv_noisy_m,
                round(SCALE_WEIGHT, 3), val_labels, probe_val_dataset,
                'Manifold vs Isotropic',
                os.path.join(manifold_dir, f"idx{target_idx}_sae_retrieval.png"),
                iso_save_dir=isotropic_dir, mani_save_dir=manifold_dir)
            # combined image grid saved in manifold_dir (covers both methods)
            save_match_grid_figure(
                target_idx, cv_orig, cv_noisy_g, cv_noisy_m,
                round(SCALE_WEIGHT, 3), val_labels, probe_val_dataset,
                manifold_dir)
            print(f"  Saved SAE retrieval + match grid for idx {target_idx} (VIZ_ONLY)", flush=True)
        else:
            print(f"  WARNING: probe_val_dataset is None — SAE figures skipped (VIZ_ONLY)", flush=True)

        print(f"  Saved viz for idx {target_idx} (VIZ_ONLY)", flush=True)
        viz_count += 1

    print(f"\nVIZ-ONLY done. Regenerated viz for {viz_count} targets.", flush=True)

# ===========================================================================
# Normal smoothing computation loop (skipped when VIZ_ONLY=1)
# ===========================================================================
for loop_i, target_idx in enumerate(TARGET_IDCS if not VIZ_ONLY else []):
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

    # Whiten original (subtract local mean before projecting — supervisor change)
    cv_whitened = (cv_orig - mean_nn) @ Vt.T / np.sqrt(ev)

    # Noise scale: alpha = sigma / sqrt(lambda_max)  (supervisor change)
    # In whitened space we add N(0, alpha^2 I); this maps to pixel-space std
    # sigma * sqrt(ev_norm_i) along each PC — ellipse matches the circle at PC1.
    lambda_max = float(ev[0])
    sqrt_lambda_max = float(np.sqrt(max(lambda_max, 1e-12)))
    alpha = SCALE_WEIGHT / sqrt_lambda_max
    print(f"  [PCA] λ_max={lambda_max:.6f}  √λ_max={sqrt_lambda_max:.6f}  α=σ/√λ_max={alpha:.6f}  (σ={SCALE_WEIGHT})", flush=True)

    # Stage 1 (n0) — choose the class to certify, paper-style
    class_counts_n0 = np.zeros(N_CLASSES, dtype=np.int64)
    for _ in range(N0_SMOOTH_SAMPLES):
        noise = np.random.normal(0, alpha, size=len(ev))
        cv_noised = cv_whitened + noise
        cv_stage0 = cv_noised @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
        pred_stage0 = classify_concept_vector(
            torch.tensor(cv_stage0, dtype=torch.float32, device=args.device),
            classifier_weights)
        class_counts_n0[pred_stage0] += 1

    # Stage 2 (n) — collect concept stats + class counts for certification
    smooth_preds = []
    class_counts_n = np.zeros(N_CLASSES, dtype=np.int64)
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
    save_viz = SAVE_VIZ and (N_VIZ < 0 or viz_count < N_VIZ)
    all_manifold_smooth_vecs = [] if save_viz else None

    for sample_i in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, alpha, size=len(ev))
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
        class_counts_n[pred_i] += 1

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

    # Downstream class certification (paper-aligned two-stage certify)
    class_cert = certify_class_votes_two_stage(
        class_counts_n0,
        class_counts_n,
        SCALE_WEIGHT,
        alpha_conf=CERTIFY_ALPHA,
        alpha_pred=CERTIFY_ALPHA,
    )
    pred_smooth = class_cert['top_class']
    n_votes = class_cert['n_votes']

    # --- Compute "average smoothed" concept vector ---
    # Re-run to get the mean smoothed vector for final concept summary
    cv_smoothed_accum = np.zeros_like(cv_orig)
    for _ in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, alpha, size=len(ev))
        cv_noised = cv_whitened + noise
        cv_smoothed_accum += cv_noised @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
    cv_smoothed_avg = cv_smoothed_accum / N_SMOOTH_SAMPLES
    final_concepts = get_top_concept_info(cv_smoothed_avg, concept_names, top_k=20)

    # --- Baseline: Isotropic Gaussian smoothing (no manifold) ---
    gauss_sigma = SCALE_WEIGHT  # same sigma as manifold for fair comparison
    gauss_counts_n0 = np.zeros(N_CLASSES, dtype=np.int64)
    for _ in range(N0_SMOOTH_SAMPLES):
        noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
        cv_gauss_n0 = cv_orig + noise
        pred_g0 = classify_concept_vector(
            torch.tensor(cv_gauss_n0, dtype=torch.float32, device=args.device),
            classifier_weights)
        gauss_counts_n0[pred_g0] += 1

    gauss_preds = []
    gauss_counts_n = np.zeros(N_CLASSES, dtype=np.int64)
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
        gauss_counts_n[pred_g] += 1
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

    gauss_class_cert = certify_class_votes_two_stage(
        gauss_counts_n0,
        gauss_counts_n,
        gauss_sigma,
        alpha_conf=CERTIFY_ALPHA,
        alpha_pred=CERTIFY_ALPHA,
    )
    pred_gauss = gauss_class_cert['top_class']
    n_votes_gauss = gauss_class_cert['n_votes']

    cv_gauss_accum = np.zeros_like(cv_orig)
    for _ in range(N_SMOOTH_SAMPLES):
        noise = np.random.normal(0, gauss_sigma, size=cv_orig.shape)
        cv_gauss_accum += cv_orig + noise
    cv_gauss_avg = cv_gauss_accum / N_SMOOTH_SAMPLES
    gauss_final_concepts = get_top_concept_info(cv_gauss_avg, concept_names, top_k=20)

    # --- Decode sanity check: noisy cv [8192] → CLIP [512] ---
    # Shows that noise in concept space produces a different point in CLIP space.
    # Uses the averaged smoothed vector as a single "virtual token" and projects
    # back to SAE input dimension (512) via the linear decoder W_dec.
    decode_info = {}
    if autoencoder is not None:
        clean_clip  = decode_cv_to_clip(cv_orig,        autoencoder, args.device)  # [512]
        m_clip      = decode_cv_to_clip(cv_smoothed_avg, autoencoder, args.device)  # [512]
        g_clip      = decode_cv_to_clip(cv_gauss_avg,   autoencoder, args.device)  # [512]
        t_clean = torch.tensor(clean_clip, dtype=torch.float32)
        t_m     = torch.tensor(m_clip,     dtype=torch.float32)
        t_g     = torch.tensor(g_clip,     dtype=torch.float32)
        sim_m = float(F.cosine_similarity(
            F.normalize(t_clean, dim=0).unsqueeze(0),
            F.normalize(t_m, dim=0).unsqueeze(0)).item())
        sim_g = float(F.cosine_similarity(
            F.normalize(t_clean, dim=0).unsqueeze(0),
            F.normalize(t_g, dim=0).unsqueeze(0)).item())
        decode_info = {
            'cosine_clean_vs_manifold': round(sim_m, 4),
            'cosine_clean_vs_gaussian': round(sim_g, 4),
        }
        print(f"    decode: cosine(clean,manifold)={sim_m:.4f}  cosine(clean,gaussian)={sim_g:.4f}  "
              f"(1.0=identical, 0.0=orthogonal)", flush=True)

    # --- Print ---
    orig_concepts = get_top_concept_info(cv_orig, concept_names, top_k=20)

    print(f"\n[{loop_i+1}/{N_TARGETS}] idx={target_idx}  "
          f"true={get_class_name(PROBE_DATASET, label_true)}  "
          f"orig={get_class_name(PROBE_DATASET, pred_orig)}  "
          f"manifold={get_class_name(PROBE_DATASET, pred_smooth)}({n_votes}/{N_SMOOTH_SAMPLES})  "
          f"gauss={get_class_name(PROBE_DATASET, pred_gauss)}({n_votes_gauss}/{N_SMOOTH_SAMPLES})  "
            f"\n    predict: m_n0={class_cert['n0_votes']}/{N0_SMOOTH_SAMPLES} p={class_cert['predict_p_value']:.4g}  "
            f"g_n0={gauss_class_cert['n0_votes']}/{N0_SMOOTH_SAMPLES} p={gauss_class_cert['predict_p_value']:.4g}"
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
            noise = np.random.normal(0, alpha, size=len(ev))
            cv_noised = cv_whitened + noise
            cv_s = cv_noised @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
            vis_noisy_points.append(cv_s)
            p = classify_concept_vector(
                torch.tensor(cv_s, dtype=torch.float32, device=args.device),
                classifier_weights)
            vis_noisy_preds.append(p)
        noisy_vis = pca_vis.transform(np.stack(vis_noisy_points))
        noisy_correct = [p == label_true for p in vis_noisy_preds]

        # Circle + Ellipse geometry overlay (pipeline artifact)
        ev_vis_norm = np.asarray(pca_vis.explained_variance_[:2], dtype=np.float64)
        if ev_vis_norm.size < 2:
            ev_vis_norm = np.asarray([1.0, 1.0], dtype=np.float64)
        ev_vis_norm = np.maximum(ev_vis_norm, 1e-12)
        ev_vis_norm = ev_vis_norm / ev_vis_norm.max()

        # ── helpers shared by both overlay functions ──────────────────────────
        def _zoom_setup(neighbors_2d, sigmas):
            pc1_std   = float(np.std(neighbors_2d[:, 0]))
            pc2_std   = float(np.std(neighbors_2d[:, 1]))
            pc1_range = float(np.max(neighbors_2d[:, 0]) - np.min(neighbors_2d[:, 0]))
            max_sigma = float(max(sigmas)) if sigmas else float(SCALE_WEIGHT)
            zoom_mid  = 0.10 * pc1_range if pc1_range > 0 else max_sigma
            zoom_tight = max_sigma
            return pc1_std, pc2_std, max_sigma, zoom_mid, zoom_tight

        def _apply_zoom_limits(ax, zr, anchor_2d, neighbors_2d):
            if zr is None:
                x_all = np.concatenate([neighbors_2d[:, 0], [anchor_2d[0]]])
                y_all = np.concatenate([neighbors_2d[:, 1], [anchor_2d[1]]])
                pad   = 0.05 * max(float(x_all.max()-x_all.min()), float(y_all.max()-y_all.min()), 1e-6)
                ax.set_xlim(float(x_all.min())-pad, float(x_all.max())+pad)
                ax.set_ylim(float(y_all.min())-pad, float(y_all.max())+pad)
            else:
                ax.set_xlim(float(anchor_2d[0])-zr, float(anchor_2d[0])+zr)
                ax.set_ylim(float(anchor_2d[1])-zr, float(anchor_2d[1])+zr)

        # ── Isotropic: 3 panels — anchor + BLUE noisy cloud + iso circle only ─
        def _save_iso_overlay(neighbors_2d, anchor_2d, sample_points_2d, sigmas, save_path):
            neighbors_2d     = np.asarray(neighbors_2d,     dtype=np.float64)
            anchor_2d        = np.asarray(anchor_2d,        dtype=np.float64)
            sample_points_2d = np.asarray(sample_points_2d, dtype=np.float64)
            if neighbors_2d.shape[0] == 0:
                return
            pc1_std, pc2_std, max_sigma, zoom_mid, zoom_tight = _zoom_setup(neighbors_2d, sigmas)
            sigma_colors = [C['iso_circle']] * max(len(sigmas), 1)
            zoom_configs = [
                (None,       f"(a) Full noise cloud   σ/std={max_sigma/(pc1_std+1e-12):.2f}"),
                (zoom_mid,   f"(b) Mid-zoom  ±{zoom_mid:.3f}"),
                (zoom_tight, f"(c) Tight zoom  ±σ={zoom_tight:.3f}"),
            ]
            fig_i, axs_i = plt.subplots(1, 3, figsize=(16, 5.5), facecolor='white')
            for ax, (zr, title) in zip(axs_i, zoom_configs):
                ax.scatter(neighbors_2d[:, 0], neighbors_2d[:, 1], c=C['knn'], s=6,
                           alpha=0.28, linewidths=0, zorder=1, label='KNN neighbors')
                if sample_points_2d.size > 0:
                    ax.scatter(sample_points_2d[:, 0], sample_points_2d[:, 1],
                               c=C['lat_iso'], s=9, alpha=0.35, linewidths=0, zorder=2,
                               label='Iso noisy samples')
                for si, sigma in enumerate(sigmas):
                    ax.add_patch(plt.Circle(
                        (float(anchor_2d[0]), float(anchor_2d[1])), float(sigma),
                        fill=False, edgecolor=sigma_colors[si], linewidth=2.0,
                        linestyle=(0, (4, 2)), alpha=0.95, zorder=4,
                        label=f'Iso circle r=σ={sigma:.3f}' if si == 0 else None,
                    ))
                ax.scatter(float(anchor_2d[0]), float(anchor_2d[1]), c=C['anchor'], s=170,
                           marker='*', edgecolors='#444', linewidths=0.8, zorder=6, label='Anchor')
                _apply_zoom_limits(ax, zr, anchor_2d, neighbors_2d)
                ax.set_aspect('equal')
                ax.set_title(title, fontsize=8.5)
                ax.set_xlabel(f'PC1 (cloud std={pc1_std:.2f})')
                ax.set_ylabel(f'PC2 (cloud std={pc2_std:.2f})')
                ax.grid(alpha=0.3)
                ax.legend(fontsize=7, loc='upper right')
            fig_i.suptitle(
                f"Isotropic Smoothing Geometry   σ={SCALE_WEIGHT}   MC={N_SMOOTH_SAMPLES}",
                fontsize=11, fontweight='bold', y=1.03,
            )
            plt.tight_layout()
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close(fig_i)

        # ── Manifold: 5 panels A-E — anchor + LIGHT-GREEN noisy cloud + circle + ellipse ─
        def _save_manifold_overlay(neighbors_2d, anchor_2d, sample_points_2d,
                                   evals_norm_2d, sigmas, save_path, evals_full_norm=None):
            neighbors_2d     = np.asarray(neighbors_2d,     dtype=np.float64)
            anchor_2d        = np.asarray(anchor_2d,        dtype=np.float64)
            sample_points_2d = np.asarray(sample_points_2d, dtype=np.float64)
            if neighbors_2d.shape[0] == 0:
                return
            pc1_std, pc2_std, max_sigma, zoom_mid, zoom_tight = _zoom_setup(neighbors_2d, sigmas)
            if evals_full_norm is not None and len(evals_full_norm) > 0:
                mid_idx      = len(evals_full_norm) // 2
                ev_mid_norm  = float(evals_full_norm[mid_idx])
                ev_last_norm = float(evals_full_norm[-1])
                n_last       = len(evals_full_norm) - 1
            else:
                mid_idx      = 0
                ev_mid_norm  = float(evals_norm_2d[-1]) if len(evals_norm_2d) > 1 else 1.0
                ev_last_norm = ev_mid_norm
                n_last       = '?'
            a_mid  = max_sigma * float(np.sqrt(max(ev_mid_norm,  0.0)))
            a_last = max_sigma * float(np.sqrt(max(ev_last_norm, 0.0)))
            zoom_configs = [
                (None,       f"(a) Full neighbourhood cloud"),
                (zoom_mid,   f"(b) Mid-zoom  ±{zoom_mid:.2f}"),
                (zoom_tight, f"(c) Ellipse axes  a₁={max_sigma:.3f}  a₂={max_sigma*float(np.sqrt(max(evals_norm_2d[1],0))) if len(evals_norm_2d)>1 else 0:.3f}"),
                (zoom_tight, f"(d) Mid axis  a_mid={a_mid:.3f}  a_mid/a₁={a_mid/(max_sigma+1e-12):.3f}"),
                (zoom_tight, f"(e) Last axis  a_last={a_last:.2e}  a_last/a₁={a_last/(max_sigma+1e-12):.2e}"),
            ]
            _n_sigmas = max(len(sigmas), 1)
            sigma_alphas = np.linspace(0.5, 0.95, _n_sigmas)
            fig_m, axs_m = plt.subplots(1, 5, figsize=(27, 5.5), facecolor='white')
            for panel_idx, (ax, (zr, panel_title)) in enumerate(zip(axs_m, zoom_configs)):
                ax.scatter(neighbors_2d[:, 0], neighbors_2d[:, 1], c=C['knn'], s=6,
                           alpha=0.28, linewidths=0, zorder=1, label='KNN neighbors')
                if sample_points_2d.size > 0:
                    ax.scatter(sample_points_2d[:, 0], sample_points_2d[:, 1],
                               c=C['mc'], s=9, alpha=0.35, linewidths=0, zorder=2,
                               label='Manifold noisy samples')
                for si, sigma in enumerate(sigmas):
                    axes_2d = axis_lengths(sigma, evals_norm_2d)
                    ax.add_patch(plt.Circle(
                        (float(anchor_2d[0]), float(anchor_2d[1])), float(sigma),
                        fill=False, edgecolor=C['iso_circle'], linewidth=1.8,
                        linestyle=(0, (4, 2)), alpha=float(sigma_alphas[si]), zorder=4,
                        label=f'Iso circle r=σ={sigma:.3f}' if si == 0 else None,
                    ))
                    if len(axes_2d) >= 2:
                        if panel_idx == 3:
                            ell_h   = float(2 * a_mid)
                            ell_lbl = f'Ellipse a1={axes_2d[0]:.4f} a_mid={a_mid:.4f}'
                        elif panel_idx == 4:
                            ell_h   = float(2 * a_last)
                            ell_lbl = f'Ellipse a1={axes_2d[0]:.4f} a_last={a_last:.4f}'
                        else:
                            ell_h   = float(2 * axes_2d[1])
                            ell_lbl = f'Ellipse a1={axes_2d[0]:.4f} a2={axes_2d[1]:.4f}'
                        ax.add_patch(Ellipse(
                            (float(anchor_2d[0]), float(anchor_2d[1])),
                            width=float(2 * axes_2d[0]), height=ell_h,
                            fill=False, edgecolor=C['mani_ellipse'], linewidth=1.8,
                            linestyle='solid', alpha=float(sigma_alphas[si]), zorder=4,
                            label=ell_lbl if si == 0 else None,
                        ))
                ax.scatter(float(anchor_2d[0]), float(anchor_2d[1]), c=C['anchor'], s=170,
                           marker='*', edgecolors='#444', linewidths=0.8, zorder=6, label='Anchor')
                _apply_zoom_limits(ax, zr, anchor_2d, neighbors_2d)
                ax.set_aspect('equal')
                ax.set_title(panel_title, fontsize=8.5)
                ax.set_xlabel(f'PC1 (cloud std={pc1_std:.2f})')
                ax.set_ylabel(f'PC2 (cloud std={pc2_std:.2f})')
                ax.grid(alpha=0.3)
                # no per-panel legend — shared legend at bottom of figure
            legend_handles_m = [
                plt.Line2D([0], [0], color=C['iso_circle'], lw=1.8, linestyle=(0, (4, 2)), label='Iso circle r=σ'),
                plt.Line2D([0], [0], color=C['mani_ellipse'], lw=1.8, linestyle='solid', label='Manifold ellipse'),
                plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=C['mc'],
                           markersize=7, label='Manifold noisy samples'),
                plt.Line2D([0], [0], marker='*', color='w', markerfacecolor=C['anchor'],
                           markeredgecolor='#444', markersize=10, label='Anchor'),
            ]
            fig_m.legend(handles=legend_handles_m, loc='lower center', ncol=4, fontsize=8,
                         frameon=True, framealpha=0.95, edgecolor='#cccccc', bbox_to_anchor=(0.5, -0.03))
            fig_m.suptitle(
                f"Manifold Smoothing Geometry   σ={SCALE_WEIGHT}\n"
                f"λ_max={lambda_max:.4f}   √λ_max={sqrt_lambda_max:.4f}   α={alpha:.4f}",
                fontsize=11, fontweight='bold', y=1.03,
            )
            plt.tight_layout(rect=[0, 0.08, 1, 1])
            plt.savefig(save_path, dpi=180, bbox_inches='tight')
            plt.close(fig_m)

        # Full-PCA normalized eigenvalues for panels D/E (mid and last PC axis)
        ev_full_norm = np.maximum(ev, 1e-30) / float(ev[0]) if ev.size > 0 else np.array([1.0])
        _save_manifold_overlay(
            X_vis, orig_vis, noisy_vis,
            ev_vis_norm, VIZ_SIGMAS,
            os.path.join(manifold_dir, f"idx{target_idx}_circle_ellipse.png"),
            evals_full_norm=ev_full_norm,
        )

        # --- Manifold: compact 1x2 — Top Activated (left) + Least Activated (right), same scale ---
        n_show = 20
        smooth_top_idxs = np.argsort(-cv_smoothed_avg)[:n_show]
        top_names       = [(concept_names[i] if concept_names else f"c_{i}") for i in smooth_top_idxs]
        top_orig_vals   = [float(cv_orig[i])        for i in smooth_top_idxs]
        top_smooth_vals = [float(cv_smoothed_avg[i]) for i in smooth_top_idxs]

        active_mask_m   = cv_orig > 1e-6
        active_idxs_m   = np.where(active_mask_m)[0]
        least_idxs_m    = sorted(active_idxs_m, key=lambda i: cv_smoothed_avg[i])[:n_show]
        least_names_m   = [(concept_names[i] if concept_names else f"c_{i}") for i in least_idxs_m]
        least_orig_m    = [float(cv_orig[i])        for i in least_idxs_m]
        least_smooth_m  = [float(cv_smoothed_avg[i]) for i in least_idxs_m]

        shared_xlim_m = max(top_orig_vals + top_smooth_vals + least_orig_m + least_smooth_m) * 1.08

        fig_m, axes_m = plt.subplots(1, 2, figsize=(18, 8))
        y = np.arange(n_show)

        ax = axes_m[0]
        ax.barh(y - 0.2, top_orig_vals[::-1],   height=0.35, color=C['overall'],  label='Original',     alpha=0.8)
        ax.barh(y + 0.2, top_smooth_vals[::-1],  height=0.35, color=C['mani_top'], label='Manifold avg', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels([n[:25] for n in top_names[::-1]], fontsize=8)
        ax.set_xlim(0, shared_xlim_m); ax.set_xlabel('Activation')
        ax.set_title(f'Top Activated  [{get_class_name(PROBE_DATASET, pred_orig)} → '
                     f'{get_class_name(PROBE_DATASET, pred_smooth)} '
                     f'{"STABLE ✓" if pred_orig == pred_smooth else "CHANGED ✗"}]',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        ax = axes_m[1]
        ax.barh(y - 0.2, least_orig_m[::-1],   height=0.35, color=C['overall'],    label='Original',     alpha=0.8)
        ax.barh(y + 0.2, least_smooth_m[::-1],  height=0.35, color=C['mani_least'], label='Manifold avg', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels([n[:25] for n in least_names_m[::-1]], fontsize=8)
        ax.set_xlim(0, shared_xlim_m); ax.set_xlabel('Activation')
        ax.set_title('Least Activated (originally active, sorted by smoothed score)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        _true_class_m  = get_class_name(PROBE_DATASET, label_true)
        _pred_smooth_class_m = get_class_name(PROBE_DATASET, pred_smooth)
        fig_m.suptitle(
            f'Manifold Smoothing   σ={SCALE_WEIGHT}\n'
            f'True: {_true_class_m}   Predicted: {_pred_smooth_class_m}',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(manifold_dir, f"idx{target_idx}_top_least_activation.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig_m)

        # SAE-space retrieval + match grid figures
        if probe_val_dataset is not None:
            cv_noisy_m_ex = cv_whitened + np.random.normal(0, alpha, size=len(ev))
            cv_noisy_m_ex = cv_noisy_m_ex @ (np.sqrt(ev)[:, None] * Vt) + mean_nn
            cv_noisy_g_ex = cv_orig + np.random.normal(0, gauss_sigma, size=cv_orig.shape)
            save_sae_retrieval_figure(
                target_idx, cv_orig, cv_noisy_g_ex, cv_noisy_m_ex,
                round(SCALE_WEIGHT, 3), val_labels, probe_val_dataset,
                'Manifold vs Isotropic',
                os.path.join(isotropic_dir, f"idx{target_idx}_sae_retrieval.png"),
                iso_save_dir=isotropic_dir, mani_save_dir=manifold_dir,
            )
            # combined image grid: original + manifold matches + iso matches
            save_match_grid_figure(
                target_idx, cv_orig, cv_noisy_g_ex, cv_noisy_m_ex,
                round(SCALE_WEIGHT, 3), val_labels, probe_val_dataset,
                manifold_dir)
            print(f"  Saved SAE retrieval + match grid for idx {target_idx}", flush=True)
        else:
            print(f"  WARNING: probe_val_dataset is None — SAE figures skipped for idx {target_idx}", flush=True)

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
            'params': {
                'K_NEIGHBORS': K_NEIGHBORS,
                'SCALE_WEIGHT': SCALE_WEIGHT,
                'N0_SMOOTH_SAMPLES': N0_SMOOTH_SAMPLES,
                'N_SMOOTH_SAMPLES': N_SMOOTH_SAMPLES,
            },
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

        _save_iso_overlay(
            X_vis,
            orig_vis,
            gauss_vis,
            VIZ_SIGMAS,
            os.path.join(isotropic_dir, f"idx{target_idx}_circle_ellipse.png"),
        )

        # --- Isotropic: compact 1x2 — Top Activated (left) + Least Activated (right), same scale ---
        n_show = 20
        gauss_top_idxs  = np.argsort(-cv_gauss_avg)[:n_show]
        g_top_names     = [(concept_names[i] if concept_names else f"c_{i}") for i in gauss_top_idxs]
        g_top_orig_vals = [float(cv_orig[i])      for i in gauss_top_idxs]
        g_top_smooth_vals = [float(cv_gauss_avg[i]) for i in gauss_top_idxs]

        active_mask_g  = cv_orig > 1e-6
        active_idxs_g  = np.where(active_mask_g)[0]
        least_idxs_g   = sorted(active_idxs_g, key=lambda i: cv_gauss_avg[i])[:n_show]
        least_names_g  = [(concept_names[i] if concept_names else f"c_{i}") for i in least_idxs_g]
        least_orig_g   = [float(cv_orig[i])      for i in least_idxs_g]
        least_smooth_g = [float(cv_gauss_avg[i]) for i in least_idxs_g]

        shared_xlim_g = max(g_top_orig_vals + g_top_smooth_vals + least_orig_g + least_smooth_g) * 1.08

        fig_g, axes_g = plt.subplots(1, 2, figsize=(18, 8))
        y = np.arange(n_show)

        ax = axes_g[0]
        ax.barh(y - 0.2, g_top_orig_vals[::-1],   height=0.35, color=C['overall'],  label='Original',      alpha=0.8)
        ax.barh(y + 0.2, g_top_smooth_vals[::-1],  height=0.35, color=C['iso_top'],  label='Isotropic avg', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels([n[:25] for n in g_top_names[::-1]], fontsize=8)
        ax.set_xlim(0, shared_xlim_g); ax.set_xlabel('Activation')
        ax.set_title(f'Top Activated  [{get_class_name(PROBE_DATASET, pred_orig)} → '
                     f'{get_class_name(PROBE_DATASET, pred_gauss)} '
                     f'{"STABLE ✓" if pred_orig == pred_gauss else "CHANGED ✗"}]',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        ax = axes_g[1]
        ax.barh(y - 0.2, least_orig_g[::-1],   height=0.35, color=C['overall'],   label='Original',      alpha=0.8)
        ax.barh(y + 0.2, least_smooth_g[::-1],  height=0.35, color=C['iso_least'], label='Isotropic avg', alpha=0.8)
        ax.set_yticks(y); ax.set_yticklabels([n[:25] for n in least_names_g[::-1]], fontsize=8)
        ax.set_xlim(0, shared_xlim_g); ax.set_xlabel('Activation')
        ax.set_title('Least Activated (originally active, sorted by smoothed score)',
                     fontsize=10, fontweight='bold')
        ax.legend(fontsize=8); ax.grid(True, axis='x', alpha=0.3)

        _true_class_g  = get_class_name(PROBE_DATASET, label_true)
        _pred_gauss_class_g = get_class_name(PROBE_DATASET, pred_gauss)
        fig_g.suptitle(
            f'Manifold Smoothing   σ={SCALE_WEIGHT}\n'
            f'True: {_true_class_g}   Predicted: {_pred_gauss_class_g}',
            fontsize=11, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(isotropic_dir, f"idx{target_idx}_top_least_activation.png"),
                    dpi=150, bbox_inches='tight')
        plt.close(fig_g)

        # SAE-space retrieval already saved above (shared manifold+iso figure)

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
            'params': {
                'GAUSS_SIGMA': round(float(gauss_sigma), 4),
                'N0_SMOOTH_SAMPLES': N0_SMOOTH_SAMPLES,
                'N_SMOOTH_SAMPLES': N_SMOOTH_SAMPLES,
            },
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
        # [REMOVED] _plot_concept_vote_histogram — not needed
        # [REMOVED] _plot_least_activated — covered by top_least_activation figure

        # ---------------------------------------------------------------
        # Concept Activation Distribution — boxplots showing spread across N samples
        # One figure per method: for each original top-K concept, show the
        # distribution of its activation value across all N smooth samples,
        # with the original activation marked as a red diamond.
        # ---------------------------------------------------------------
        def _plot_activation_distribution(all_smooth_vecs, orig_top_idxs_arr, cv_orig,
                                          concept_votes, concept_names_list, orig_top_set,
                                          method_name, save_path, target_idx, n_samples,
                                          n_show=10):
            """3-row boxplot: top-10 concepts, least-10, new-10. Purple theme."""
            smooth_mat = np.stack(all_smooth_vecs)   # [N, 8192]

            # Row 1: top-10 by original activation
            sorted_top = sorted(orig_top_idxs_arr, key=lambda i: cv_orig[i], reverse=True)[:n_show]

            # Row 2: least active — originally active but lowest activation
            all_active = np.where(cv_orig > 1e-6)[0]
            least_candidates = sorted([i for i in all_active if i not in orig_top_set],
                                      key=lambda i: cv_orig[i])[:n_show]
            if len(least_candidates) < n_show:
                weak = sorted(orig_top_idxs_arr, key=lambda i: cv_orig[i])
                for i in weak:
                    if i not in least_candidates:
                        least_candidates.append(i)
                    if len(least_candidates) >= n_show:
                        break

            # Row 3: new concepts — appeared after smoothing, not in original top
            new_concept_idxs = [c[0] for c in concept_votes.most_common()
                                 if c[0] not in orig_top_set][:n_show]

            def _draw_row(ax, concept_idxs, row_color, row_label):
                if not concept_idxs:
                    ax.text(0.5, 0.5, 'None', ha='center', va='center',
                            transform=ax.transAxes, fontsize=9)
                    ax.set_title(row_label, fontsize=10, fontweight='bold')
                    return
                c_names = [(concept_names_list[i] if concept_names_list else f"c{i}")[:25]
                           for i in concept_idxs]
                data      = [smooth_mat[:, i].tolist() for i in concept_idxs]
                orig_vals = [float(cv_orig[i]) for i in concept_idxs]

                bp = ax.boxplot(data, positions=range(len(c_names)), widths=0.55,
                                patch_artist=True, showfliers=False,
                                medianprops=dict(color='white', linewidth=1.5))
                for patch in bp['boxes']:
                    patch.set_facecolor(row_color)
                    patch.set_alpha(0.7)
                ax.scatter(range(len(c_names)), orig_vals, c='#e41a1c', s=60,
                           marker='D', zorder=5, label='Original', edgecolors='darkred',
                           linewidths=0.8)
                for j, d in enumerate(data):
                    jitter = np.random.normal(0, 0.07, size=len(d))
                    ax.scatter(np.full(len(d), j) + jitter, d,
                               c=row_color, s=3, alpha=0.12, zorder=2)
                ax.set_xticks(range(len(c_names)))
                ax.set_xticklabels(c_names, rotation=40, ha='right', fontsize=8)
                ax.set_ylabel('Activation', fontsize=8)
                ax.set_title(row_label, fontsize=10, fontweight='bold')
                ax.legend(fontsize=8, loc='upper right')
                ax.grid(True, axis='y', alpha=0.2, linestyle='--')
                ax.axhline(y=0, color='gray', ls='-', alpha=0.2)
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)

            fig, axes = plt.subplots(3, 1, figsize=(max(12, n_show * 0.9), 15))
            _draw_row(axes[0], sorted_top,       C['dist_top'],   f'Top {n_show} concepts')
            _draw_row(axes[1], least_candidates, C['dist_least'], f'Least {n_show} concepts')
            _draw_row(axes[2], new_concept_idxs, C['dist_new'],   f'New {n_show} concepts (emerged after smoothing)')

            fig.suptitle(
                f'{method_name}   idx={target_idx}   '
                f'true={get_class_name(PROBE_DATASET, label_true)}',
                fontsize=11, fontweight='bold', y=1.01)
            plt.tight_layout()
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            plt.close(fig)

        _plot_activation_distribution(
            all_manifold_smooth_vecs, orig_top_idxs, cv_orig,
            all_concept_votes_manifold, concept_names, orig_top,
            'Manifold Smoothing',
            os.path.join(manifold_dir, f"idx{target_idx}_activation_dist.png"),
            target_idx, N_SMOOTH_SAMPLES)

        _plot_activation_distribution(
            all_gauss_smooth_vecs, orig_top_idxs, cv_orig,
            all_concept_votes_gaussian, concept_names, orig_top,
            'Isotropic Smoothing',
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

    # Volume framework: certified radii + geometry-first sigma-based diagnostics
    r_iso = gauss_class_cert['certified_radius']
    r_mani = class_cert['certified_radius']
    # Filter eigenvalues to non-trivial components
    ev_positive = ev[ev > 1e-10]
    CONCEPT_DIM = len(cv_orig)
    volumes = compute_volumes(r_iso, r_mani, ev_positive, D=CONCEPT_DIM, sigma=SCALE_WEIGHT)

    result_row = {
        'idx': target_idx,
        'img_path': str(img_path) if img_path else None,
        'label_true': label_true,
        'pred_orig': pred_orig,
        'n0_samples': N0_SMOOTH_SAMPLES,
        'n_samples': N_SMOOTH_SAMPLES,
        # === Concept-level certification (manifold) ===
        'mean_concept_survival_manifold': mean_concept_survival,
        'n_concepts_certified_manifold': n_concepts_certified,
        'min_concept_radius_manifold': min_concept_radius,
        'mean_concept_radius_manifold': mean_concept_radius,
        'concept_survival_manifold': concept_survival_named,
        'overlap_manifold': round(float(np.mean(smooth_concepts_overlap)), 4),
        # === Downstream class certification (manifold) ===
        'pred_manifold': pred_smooth,
        'pred_manifold_stage2': class_cert.get('top_class_stage2', pred_smooth),
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
        'pred_gaussian_stage2': gauss_class_cert.get('top_class_stage2', pred_gauss),
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
n = len(results)

# Helper: filter out non-finite values for safe mean/median
def _finite(vals):
    return [v for v in vals if np.isfinite(v) and abs(v) < 1e30]

def _safe_mean(vals):
    fv = _finite(vals)
    return np.mean(fv) if fv else float('nan')

def _safe_median(vals):
    fv = _finite(vals)
    return np.median(fv) if fv else float('nan')

# ==========================================================================
# TABLE 1: DOWNSTREAM ACCURACY (no certification, just prediction quality)
# ==========================================================================
print(f"\n{'='*80}")
print("TABLE 1: DOWNSTREAM ACCURACY")
print(f"{'='*80}")

orig_correct = sum(1 for r in results if r['pred_orig'] == r['label_true'])
m_correct = sum(1 for r in results if r['pred_manifold'] == r['label_true'])
g_correct = sum(1 for r in results if r['pred_gaussian'] == r['label_true'])
m_preserves = sum(1 for r in results if r['pred_manifold'] == r['pred_orig'])
g_preserves = sum(1 for r in results if r['pred_gaussian'] == r['pred_orig'])

print(f"{'Method':<18s} {'Accuracy':<16s} {'Preserves Orig':<18s}")
print(f"{'-'*52}")
print(f"{'Original':<18s} {orig_correct}/{n} ({orig_correct/n*100:.1f}%){'':>4s} {'—':<18s}")
print(f"{'Manifold (σ={SCALE_WEIGHT})':<18s} {m_correct}/{n} ({m_correct/n*100:.1f}%){'':>4s} "
      f"{m_preserves}/{n} ({m_preserves/n*100:.1f}%)")
print(f"{'Gaussian (σ={SCALE_WEIGHT})':<18s} {g_correct}/{n} ({g_correct/n*100:.1f}%){'':>4s} "
      f"{g_preserves}/{n} ({g_preserves/n*100:.1f}%)")

# ==========================================================================
# TABLE 2: CERTIFIED DOWNSTREAM ACCURACY (paper-aligned two-stage)
# ==========================================================================
print(f"\n{'='*80}")
print(f"TABLE 2: CERTIFIED DOWNSTREAM ACCURACY (paper-aligned two-stage, α={CERTIFY_ALPHA})")
print(f"{'='*80}")

def _class_stats(results, method):
    """Extract class certification stats."""
    cert_key = f'class_cert_{method}'
    pred_key = f'pred_{method}'

    n_certified = 0          # not abstained
    n_correct_certified = 0  # certified AND matches true label
    n_stable_certified = 0   # certified AND matches original pred
    n_abstained = 0
    radii = []
    for r in results:
        cc = r.get(cert_key, {})
        if cc.get('abstained', True):
            n_abstained += 1
        else:
            n_certified += 1
            rad = cc.get('certified_radius', 0)
            if np.isfinite(rad):
                radii.append(rad)
            if r[pred_key] == r['label_true']:
                n_correct_certified += 1
            if r[pred_key] == r['pred_orig']:
                n_stable_certified += 1
    return {
        'n_certified': n_certified,
        'n_correct_certified': n_correct_certified,
        'n_stable_certified': n_stable_certified,
        'certified_accuracy': n_correct_certified / n if n else 0,
        'certified_stability': n_stable_certified / n if n else 0,
        'abstain_rate': n_abstained / n if n else 0,
        'mean_radius': np.mean(radii) if radii else 0,
        'median_radius': np.median(radii) if radii else 0,
        'max_radius': max(radii) if radii else 0,
    }

m_cls = _class_stats(results, 'manifold')
g_cls = _class_stats(results, 'gaussian')

print(f"{'Method':<18s} {'CertAcc↑':<12s} {'CertStab↑':<12s} {'Abstain↓':<10s} "
      f"{'#Cert':<8s} {'MeanR↑':<10s} {'MedR':<10s}")
print(f"{'-'*80}")
for label, s in [('Manifold', m_cls), ('Gaussian', g_cls)]:
    print(f"{label+f' (σ={SCALE_WEIGHT})':<18s} {s['certified_accuracy']:<12.3f} "
          f"{s['certified_stability']:<12.3f} {s['abstain_rate']:<10.3f} "
          f"{str(s['n_certified'])+'/'+str(n):<8s} "
          f"{s['mean_radius']:<10.4f} {s['median_radius']:<10.4f}")

# ==========================================================================
# TABLE 3: CONCEPT-LEVEL CERTIFICATION (concept space radius)
# ==========================================================================
print(f"\n{'='*80}")
print(f"TABLE 3: CONCEPT CERTIFICATION (top-{orig_top_k} concepts, α={CERTIFY_ALPHA})")
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

print(f"{'Method':<18s} {'Survival↑':<10s} {'#Cert/'+str(orig_top_k)+'↑':<10s} {'Overlap↑':<10s} "
      f"{'MeanR↑':<12s} {'MinR':<12s}")
print(f"{'-'*72}")
print(f"{'Manifold':<18s} {np.mean(m_surv):<10.3f} {np.mean(m_cert):<10.1f} {np.mean(m_overlap):<10.3f} "
      f"{_safe_mean(m_mean_r):<12.4f} {_safe_mean(m_min_r):<12.4f}")
print(f"{'Gaussian':<18s} {np.mean(g_surv):<10.3f} {np.mean(g_cert):<10.1f} {np.mean(g_overlap):<10.3f} "
      f"{_safe_mean(g_mean_r):<12.4f} {_safe_mean(g_min_r):<12.4f}")

# ==========================================================================
# TABLE 4: CONCEPT STABILITY SCORES
# ==========================================================================
print(f"\n{'='*80}")
print("TABLE 4: CONCEPT STABILITY SCORES")
print(f"{'='*80}")
score_keys = ['concept_fidelity', 'rank_correlation', 'concept_drift',
              'spurious_act_rate', 'act_drop_score', 'act_gain_score']
score_labels = ['Fidelity(r)↑', 'RankCorr(τ)↑', 'Drift(L2)↓', 'SAR↓', 'ADS↓', 'AGS↓']
print(f"{'Method':<18s} " + " ".join(f"{l:<13s}" for l in score_labels))
print(f"{'-'*98}")
for method, label in [('manifold', 'Manifold'), ('gaussian', 'Gaussian')]:
    vals = []
    for k in score_keys:
        v = [r.get(f'scores_{method}', {}).get(k, 0) for r in results]
        vals.append(np.mean(v))
    print(f"{label:<18s} " + " ".join(f"{v:<13.4f}" for v in vals))

# ==========================================================================
# TABLE 5: CERTIFIED VOLUME (4 quantities, log-space)
# ==========================================================================
print(f"\n{'='*80}")
print("TABLE 5: CERTIFIED VOLUME (log-space, finite samples only)")
print(f"{'='*80}")
vol_keys = ['log_vol_iso_D', 'log_vol_iso_k', 'log_vol_mani_pred', 'log_vol_mani_actual']
vol_labels = ['Qty1: Iso Ball (D)',
              'Qty2: Iso Ball (k)',
              'Qty3: Mani w/ iso-r',
              'Qty4: Mani Ellipsoid']
vol_descriptions = [
    'C_D · r_iso^D          (ambient isotropic ball)',
    'C_k · r_iso^k          (projected isotropic ball)',
    'C_k · r_iso^k · √det(Λ) (manifold-aware, iso radius)',
    'C_k · r_mani^k · √det(Λ) (manifold ellipsoid, mani radius)',
]

# Collect r_iso and r_mani
r_isos = [r.get('volumes', {}).get('r_iso', 0) for r in results]
r_manis = [r.get('volumes', {}).get('r_mani', 0) for r in results]

print(f"\n  σ = {SCALE_WEIGHT}")
print(f"  Sampling = n0={N0_SMOOTH_SAMPLES}, n={N_SMOOTH_SAMPLES}")
print(f"  r_iso  (Gaussian cert radius): mean={_safe_mean(r_isos):.4f}, "
      f"median={_safe_median(r_isos):.4f}, >0: {sum(1 for r in r_isos if r > 0)}/{n}")
print(f"  r_mani (Manifold cert radius): mean={_safe_mean(r_manis):.4f}, "
      f"median={_safe_median(r_manis):.4f}, >0: {sum(1 for r in r_manis if r > 0)}/{n}")
mean_k = np.mean([r.get('n_eigenvalues', 0) for r in results])
_concept_dim_summary = int(val_concept_vectors.shape[1])
print(f"  Effective manifold dim: mean k={mean_k:.1f} / D={_concept_dim_summary}")

print(f"\n{'Quantity':<25s} {'Mean':<12s} {'Median':<12s} {'Std':<12s} {'#Finite':<10s}")
print(f"{'-'*71}")
for vk, vl in zip(vol_keys, vol_labels):
    vals = [r.get('volumes', {}).get(vk, -np.inf) for r in results]
    fv = _finite(vals)
    n_fin = len(fv)
    if fv:
        print(f"{vl:<25s} {np.mean(fv):<12.2f} {np.median(fv):<12.2f} {np.std(fv):<12.2f} {n_fin}/{n}")
    else:
        print(f"{vl:<25s} {'N/A':<12s} {'N/A':<12s} {'N/A':<12s} {n_fin}/{n}")

print(f"\nNote: Qty1-3 are -Inf when r_iso=0 (Gaussian abstained).")
print(f"      Qty4 is -Inf when r_mani=0 (Manifold abstained).")
print(f"      Means are computed over finite values only.")

geo_iso_vals = [r.get('volumes', {}).get('log_vol_geo_iso_k', -np.inf) for r in results]
geo_mani_vals = [r.get('volumes', {}).get('log_vol_geo_mani', -np.inf) for r in results]
geo_ratio_vals = [r.get('volumes', {}).get('log_geo_ratio', -np.inf) for r in results]
anisotropy_vals = [r.get('volumes', {}).get('anisotropy_ratio', 0.0) for r in results if r.get('volumes', {}).get('anisotropy_ratio', 0.0) > 0]
effective_rank_vals = [r.get('volumes', {}).get('eigen_effective_rank', 0.0) for r in results if r.get('volumes', {}).get('eigen_effective_rank', 0.0) > 0]
axis_length_rows = [r.get('volumes', {}).get('axis_lengths', []) for r in results if r.get('volumes', {}).get('axis_lengths')]

print(f"\n{'='*80}")
print("TABLE 6: GEOMETRY-FIRST MANIFOLD METRICS (σ-based, normalized eigenvalues)")
print(f"{'='*80}")
print("  λ̃_i = λ_i / λ_max")
print("  V_iso,geo = C_k · σ^k")
print("  V_mani,geo = C_k · σ^k · √det(Λ̃)")
print("  a_i = σ · √λ̃_i")

geo_iso_fin = _finite(geo_iso_vals)
geo_mani_fin = _finite(geo_mani_vals)
geo_ratio_fin = _finite(geo_ratio_vals)
if geo_iso_fin:
    print(f"  log V_iso,geo:   mean={np.mean(geo_iso_fin):.2f}, median={np.median(geo_iso_fin):.2f}")
if geo_mani_fin:
    print(f"  log V_mani,geo:  mean={np.mean(geo_mani_fin):.2f}, median={np.median(geo_mani_fin):.2f}")
if geo_ratio_fin:
    print(f"  log geo ratio:   mean={np.mean(geo_ratio_fin):.2f}, median={np.median(geo_ratio_fin):.2f}")
if anisotropy_vals:
    print(f"  anisotropy:      mean={np.mean(anisotropy_vals):.2f}, median={np.median(anisotropy_vals):.2f}")
if effective_rank_vals:
    print(f"  effective rank:  mean={np.mean(effective_rank_vals):.2f}, median={np.median(effective_rank_vals):.2f}")
if axis_length_rows:
    max_len = max(len(row) for row in axis_length_rows)
    axis_mat = np.full((len(axis_length_rows), max_len), np.nan, dtype=np.float64)
    for row_idx, row in enumerate(axis_length_rows):
        axis_mat[row_idx, :len(row)] = row
    mean_axis = np.nanmean(axis_mat, axis=0)
    shown = min(5, len(mean_axis))
    show_axes = ", ".join(f"a{i+1}={mean_axis[i]:.4f}" for i in range(shown))
    print(f"  mean axis lengths (first {shown}): {show_axes}")

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
