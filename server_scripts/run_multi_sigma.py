"""
Run smoothing experiments across multiple sigma values.
Creates per-sigma result folders and a combined summary.

Usage:
  python server_scripts/run_multi_sigma.py

  # or as SLURM array job:
  sbatch server_scripts/multi_sigma_job.sh
"""

import subprocess
import sys
import os
import json
import numpy as np
from pathlib import Path

# ===========================================================================
# Config
# ===========================================================================
SIGMAS = [0.25, 0.50, 0.70, 0.75, 1.00]
SCRIPT = os.path.join(os.path.dirname(__file__), 'run_smoothing.py')
BASE_SAVE_DIR = os.path.join(os.path.dirname(__file__), '..', 'smoothing_data')


def run_sigma(sigma):
    """Run smoothing for a single sigma value."""
    env = os.environ.copy()
    env['CFM_SIGMA'] = str(sigma)
    print(f"\n{'='*70}")
    print(f"Running sigma={sigma}")
    print(f"{'='*70}\n")
    result = subprocess.run(
        [sys.executable, SCRIPT],
        env=env,
        cwd=os.path.dirname(SCRIPT),
    )
    if result.returncode != 0:
        print(f"WARNING: sigma={sigma} failed with return code {result.returncode}")
    return result.returncode == 0


def collect_summaries():
    """Collect results across all sigmas into a combined summary."""
    combined = []
    for sigma in SIGMAS:
        sigma_dir = os.path.join(BASE_SAVE_DIR, f"sigma_{sigma:.2f}")
        jsonl_path = os.path.join(sigma_dir, "smoothing_results_imagenet.jsonl")
        if not os.path.exists(jsonl_path):
            print(f"  sigma={sigma}: no results found")
            continue
        results = []
        with open(jsonl_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        results.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
        if not results:
            continue

        # Aggregate
        r_iso_vals = [r.get('volumes', {}).get('r_iso', 0) for r in results]
        r_mani_vals = [r.get('volumes', {}).get('r_mani', 0) for r in results]
        vol_keys = ['log_vol_iso_D', 'log_vol_iso_k', 'log_vol_mani_pred', 'log_vol_mani_actual']
        vol_means = {}
        for vk in vol_keys:
            vals = [r.get('volumes', {}).get(vk, -np.inf) for r in results]
            finite = [v for v in vals if v > -1e30]
            vol_means[vk] = float(np.mean(finite)) if finite else None

        cert_rate_mani = np.mean([not r.get('class_cert_manifold', {}).get('abstained', True) for r in results])
        cert_rate_gauss = np.mean([not r.get('class_cert_gaussian', {}).get('abstained', True) for r in results])

        combined.append({
            'sigma': sigma,
            'n_samples': len(results),
            'mean_r_iso': round(float(np.mean(r_iso_vals)), 6),
            'mean_r_mani': round(float(np.mean(r_mani_vals)), 6),
            'cert_rate_manifold': round(cert_rate_mani, 4),
            'cert_rate_gaussian': round(cert_rate_gauss, 4),
            **{k: round(v, 2) if v is not None else None for k, v in vol_means.items()},
        })

    # Save combined
    out_path = os.path.join(BASE_SAVE_DIR, "multi_sigma_summary.json")
    with open(out_path, 'w') as f:
        json.dump(combined, f, indent=2)
    print(f"\nCombined summary saved to: {out_path}")

    # Print table
    print(f"\n{'='*90}")
    print(f"{'sigma':<8s} {'r_iso':<10s} {'r_mani':<10s} {'cert_m':<10s} {'cert_g':<10s} "
          f"{'logV_isoD':<12s} {'logV_isok':<12s} {'logV_pred':<12s} {'logV_mani':<12s}")
    print(f"{'-'*90}")
    for row in combined:
        print(f"{row['sigma']:<8.2f} {row['mean_r_iso']:<10.4f} {row['mean_r_mani']:<10.4f} "
              f"{row['cert_rate_manifold']:<10.3f} {row['cert_rate_gaussian']:<10.3f} "
              f"{str(row.get('log_vol_iso_D', 'N/A')):<12s} "
              f"{str(row.get('log_vol_iso_k', 'N/A')):<12s} "
              f"{str(row.get('log_vol_mani_pred', 'N/A')):<12s} "
              f"{str(row.get('log_vol_mani_actual', 'N/A')):<12s}")

    return combined


if __name__ == '__main__':
    # Check if running a single sigma from env (SLURM array mode)
    env_sigma = os.environ.get('CFM_SIGMA')
    if env_sigma:
        sigma = float(env_sigma)
        run_sigma(sigma)
    else:
        # Run all sigmas sequentially
        for sigma in SIGMAS:
            os.environ['CFM_SIGMA'] = str(sigma)
            run_sigma(sigma)
        # Collect
        collect_summaries()
