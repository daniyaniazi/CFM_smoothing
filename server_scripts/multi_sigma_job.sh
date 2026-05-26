#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 12:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing_sigma-%a-%j.out
#SBATCH -e /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing_sigma-%a-%j.err
#SBATCH -J cfm-multi-sigma
#SBATCH --array=0-19

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/CFM_smoothing"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]; then
    . "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate cfm-env
fi

# Map SLURM_ARRAY_TASK_ID to sigma values (20 values from 0.01 to 0.25)
SIGMAS=(0.010 0.023 0.035 0.048 0.061 0.073 0.086 0.099 0.111 0.124 0.136 0.149 0.162 0.174 0.187 0.200 0.212 0.225 0.237 0.250)
export CFM_SIGMA=${SIGMAS[$SLURM_ARRAY_TASK_ID]}
export CFM_N_SAMPLES=100

echo "================================================"
echo "CFM Smoothing — sigma=$CFM_SIGMA, N=$CFM_N_SAMPLES"
echo "Array task: $SLURM_ARRAY_TASK_ID / Job: $SLURM_JOB_ID"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python server_scripts/run_smoothing.py

echo "Done: $(date)"
