#!/bin/bash

#SBATCH --account=def-sponsor00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/project/def-sponsor00/remilalonde/logs/convlstm_%j.out
#SBATCH --error=/project/def-sponsor00/remilalonde/logs/convlstm_%j.err


# ============================================================
# Environment setup — mirrors the working 2020/CGNet pipeline
# ============================================================

echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

module --force purge
module load python/3.10
module load cuda
module load netcdf
module load hdf5

source ~/venv_clean/bin/activate

echo "Python: $(which python)"
python --version

# ============================================================
# Run training
# ============================================================

PROJECT_DIR=/home/remilalonde/projects/def-sponsor00/remilalonde
cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export TRAIN_FOLDER=/project/def-sponsor00/shared_CN_B/climatenet_engineered/train

echo "Starting ConvLSTM training..."
python ConvLSTM/train.py
echo "Done at $(date)"
