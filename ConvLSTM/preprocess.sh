#!/bin/bash

#SBATCH --account=def-sponsor00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/remilalonde/Project/IFT6759-Project/logs/preprocess_%j.out
#SBATCH --error=/home/remilalonde/Project/IFT6759-Project/logs/preprocess_%j.err


echo "Preprocess job started on $(hostname) at $(date)"
echo "Job ID: $SLURM_JOB_ID"

module load StdEnv/2020 gcc/9.3.0 python/3.10 hdf5/1.12.1

PROJECT_DIR=/home/remilalonde/Project/IFT6759-Project
source $PROJECT_DIR/.venv/bin/activate

cd $PROJECT_DIR

export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
export HDF5_USE_FILE_LOCKING=FALSE
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export TRAIN_FOLDER=/project/def-sponsor00/shared_CN_B/climatenet_engineered/train
export NPY_FOLDER=$PROJECT_DIR/data/climatenet_npy/train
export CHANNELS=TMQ,U850,V850,PSL

python data/preprocess.py
echo "Done at $(date)"
