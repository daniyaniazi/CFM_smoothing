#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing_sigma-%a-%j.out
#SBATCH -e /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing_sigma-%a-%j.err
#SBATCH -J cfm-multi-sigma
#SBATCH --array=0-29

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/CFM_smoothing"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]; then
    . "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate cfm-env
fi

# 30 sigma values linearly spaced from 0.003 to 1.000
SIGMAS=(0.003 0.037 0.072 0.106 0.141 0.175 0.210 0.244 0.279 0.313
        0.348 0.382 0.417 0.451 0.486 0.520 0.555 0.589 0.624 0.658
        0.693 0.727 0.762 0.796 0.831 0.865 0.900 0.934 0.969 1.000)
export CFM_SIGMA=${SIGMAS[$SLURM_ARRAY_TASK_ID]}
export CFM_N0_SAMPLES=50
export CFM_N_SAMPLES=500
export CFM_SAVE_VIZ=${CFM_SAVE_VIZ:-1}
export CFM_N_VIZ=${CFM_N_VIZ:-10}
export CFM_VIZ_SIGMAS=${CFM_VIZ_SIGMAS:-$CFM_SIGMA}

echo "================================================"
echo "CFM Smoothing — sigma=$CFM_SIGMA, N=$CFM_N_SAMPLES"
echo "Viz: SAVE=$CFM_SAVE_VIZ, N_VIZ=$CFM_N_VIZ, VIZ_SIGMAS=$CFM_VIZ_SIGMAS"
echo "Array task: $SLURM_ARRAY_TASK_ID / Job: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python server_scripts/run_smoothing.py

echo "Done: $(date)"
