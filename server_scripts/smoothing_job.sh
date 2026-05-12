#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 04:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=8G
#SBATCH -o /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing-%j.out
#SBATCH -e /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/smoothing-%j.err
#SBATCH -J cfm-smoothing

set -euo pipefail

PROJECT_ROOT="/BS/dniazi_thesis/work/CFM_smoothing"
cd "$PROJECT_ROOT"
mkdir -p output/slurm
export PYTHONPATH="$PROJECT_ROOT:${PYTHONPATH:-}"

if [ -f "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh" ]; then
    . "/BS/dniazi_thesis/work/miniforge3_new/etc/profile.d/conda.sh"
    conda activate cfm-env
fi

echo "================================================"
echo "CFM Manifold Smoothing in Concept Space"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python server_scripts/run_smoothing.py

echo "Done: $(date)"
