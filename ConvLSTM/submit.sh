#!/bin/bash

#SBATCH --account=def-sponsor00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --job-name=convlstm
#SBATCH --output=/home/remilalonde/Project/IFT6759-Project/logs/%x_%j.out
#SBATCH --error=/home/remilalonde/Project/IFT6759-Project/logs/%x_%j.err


echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

module --force purge

module load StdEnv/2020 gcc/9.3.0 cuda/11.4 python/3.10 mpi4py/3.1.3

PROJECT_DIR=/home/remilalonde/Project/IFT6759-Project
source $PROJECT_DIR/.venv/bin/activate

echo "Python: $(which python)"
python --version

cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TRAIN_FOLDER=/project/def-sponsor00/shared_CN_B/climatenet_engineered/train
export CHECKPOINT_DIR=/home/remilalonde/Project/IFT6759-Project/checkpoints/${SLURM_JOB_NAME}

echo "Starting ConvLSTM training..."
python ConvLSTM/train.py
echo "Done at $(date)"
