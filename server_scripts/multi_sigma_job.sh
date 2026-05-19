#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing_sigma-%a-%j.out
#SBATCH -e /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing_sigma-%a-%j.err
#SBATCH -J cfm-multi-sigma
#SBATCH --array=0-4

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/CFM_smoothing"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]; then
    . "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate cfm-env
fi

# Map SLURM_ARRAY_TASK_ID to sigma values
SIGMAS=(0.25 0.50 0.70 0.75 1.00)
export CFM_SIGMA=${SIGMAS[$SLURM_ARRAY_TASK_ID]}

echo "================================================"
echo "CFM Smoothing — sigma=$CFM_SIGMA"
echo "Array task: $SLURM_ARRAY_TASK_ID / Job: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python server_scripts/run_smoothing.py

echo "Done: $(date)"
