#!/usr/bin/env bash
#SBATCH -p gpu20
#SBATCH -t 06:00:00
#SBATCH --gres gpu:1
#SBATCH -c 8
#SBATCH --mem-per-cpu=16G
#SBATCH -o /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/build_knn-%j.out
#SBATCH -e /BS/dniazi_thesis/work/CFM_smoothing/output/slurm/build_knn-%j.err
#SBATCH -J cfm-build-knn

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
echo "CFM Build KNN Index"
echo "Running on: $(hostname)"
echo "GPU: ${CUDA_VISIBLE_DEVICES:-unset}"
echo "Start: $(date)"
echo "================================================"

python server_scripts/build_knn_index.py --force

echo "Done: $(date)"
