#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --job-name=hybrid
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=/home/remilalonde/Project/IFT6759-Project/hybrid/logs/hybrid_%j.out
#SBATCH --error=/home/remilalonde/Project/IFT6759-Project/hybrid/logs/hybrid_%j.err

echo "Starting hybrid CGNet+ConvLSTM job $SLURM_JOB_ID"

module --force purge
module load StdEnv/2023
module load gcc/12.3 openmpi/4.1.5 mpi4py/3.1.6
module load python/3.10.13

source /home/remilalonde/Project/IFT6759-Project/.venv/bin/activate

cd /home/remilalonde/Project/IFT6759-Project

export TRAIN_FOLDER=/project/def-sponsor00/shared_CN_B/climatenet_engineered/train
export CHECKPOINT_DIR=/home/remilalonde/Project/IFT6759-Project/checkpoints/hybrid
export CGNET_WEIGHTS=/home/remilalonde/Project/IFT6759-Project/climatenet/outputs/cgnet_4ch/trained_cgnet/weights.pth
export TEST_FOLDER=/project/def-sponsor00/shared_CN_B/climatenet_engineered/test

python -m hybrid.train

echo "Job finished"
