#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 24:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing_grid-%a-%j.out
#SBATCH -e /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing_grid-%a-%j.err
#SBATCH -J cfm-grid
#SBATCH --array=0-24

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/CFM_smoothing"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]; then
    . "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate cfm-env
fi

# Full grid: 5 sigmas x 5 N values = 25 jobs
SIGMAS=(0.25 0.50 0.70 0.75 1.00)
N_VALUES=(100 300 500 700 1000)

# Map array task ID to (sigma_idx, n_idx)
SIGMA_IDX=$((SLURM_ARRAY_TASK_ID / 5))
N_IDX=$((SLURM_ARRAY_TASK_ID % 5))

export CFM_SIGMA=${SIGMAS[$SIGMA_IDX]}
export CFM_N_SAMPLES=${N_VALUES[$N_IDX]}

echo "================================================"
echo "CFM Smoothing — sigma=$CFM_SIGMA, N=$CFM_N_SAMPLES"
echo "Grid task: $SLURM_ARRAY_TASK_ID (sigma_idx=$SIGMA_IDX, n_idx=$N_IDX)"
echo "Job: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python server_scripts/run_smoothing.py

echo "Done: $(date)"
