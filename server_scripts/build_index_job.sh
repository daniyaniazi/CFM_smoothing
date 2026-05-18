#!/bin/bash
#SBATCH --job-name=build_knn_index
#SBATCH --output=logs/build_knn_%j.out
#SBATCH --error=logs/build_knn_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=06:00:00

# --- Setup ---
cd $SLURM_SUBMIT_DIR
mkdir -p logs

# Activate environment (adjust to your setup)
source activate cfm
# Alternative: conda activate cfm
# Alternative: source /path/to/venv/bin/activate

echo "============================================"
echo "Job ID:        $SLURM_JOB_ID"
echo "Node:          $SLURM_NODELIST"
echo "GPUs:          $CUDA_VISIBLE_DEVICES"
echo "Start time:    $(date)"
echo "Working dir:   $(pwd)"
echo "============================================"

# Run with --force to rebuild from scratch
python server_scripts/build_knn_index.py --force

echo "============================================"
echo "End time:      $(date)"
echo "Exit code:     $?"
echo "============================================"
