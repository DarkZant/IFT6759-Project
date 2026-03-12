#!/bin/bash
# ============================================================
# SLURM job script for CalculQuébec (Narval or Béluga)
# Submit with:  sbatch submit.sh
# Monitor with: squeue -u $USER
# Cancel with:  scancel <JOBID>
# ============================================================

#SBATCH --job-name=convlstm_climatenet
#SBATCH --account=YOUR_ACCOUNT          # e.g. def-yourpi or rrg-yourpi
#SBATCH --output=logs/%x_%j.out         # stdout → logs/convlstm_<jobid>.out
#SBATCH --error=logs/%x_%j.err          # stderr → logs/convlstm_<jobid>.err
#SBATCH --time=12:00:00                 # max wall time (HH:MM:SS) — adjust per run

# --- GPU ---
#SBATCH --gres=gpu:1                    # 1 GPU (A100 on Narval, V100 on Béluga)

# --- CPU & memory ---
#SBATCH --cpus-per-task=4              # CPU cores for DataLoader workers
#SBATCH --mem=48G                       # RAM — 768x1152 grids are large


# ============================================================
# Environment setup
# ============================================================

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

# Load modules — adjust versions to what is available on your cluster
# Check available: module avail python  /  module avail cuda
module load python/3.10
module load cuda/11.8             # use cuda/12.1 on Narval if available

# Activate your virtualenv (must be pre-built — see setup steps below)
source $SLURM_SUBMIT_DIR/.venv/bin/activate

# Go to project directory
cd $SLURM_SUBMIT_DIR

# ============================================================
# Run training
# ============================================================

echo "Starting training..."
python ConvLSTM/train.py
echo "Done at $(date)"
