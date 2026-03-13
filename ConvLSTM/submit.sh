#!/bin/bash

#SBATCH --account=def-sponsor00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/home/remilalonde/Project/IFT6759-Project/logs/convlstm_%j.out
#SBATCH --error=/home/remilalonde/Project/IFT6759-Project/logs/convlstm_%j.err


echo "Job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

module --force purge
module load python/3.10
module load cuda
module load netcdf
module load hdf5

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
export NPY_FOLDER=$PROJECT_DIR/data/climatenet_npy/train

echo "Starting ConvLSTM training..."
python ConvLSTM/train.py
echo "Done at $(date)"
