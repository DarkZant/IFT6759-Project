#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=/project/def-sponsor00/remilalonde/logs/climatenet_%j.out

echo "Hostname:"
hostname

echo "CPU flags:"
lscpu | grep Flags

echo "Starting training job $SLURM_JOB_ID"

module --force purge

# Charger MINIMUM strict
module load python/3.10
module load cuda
module load netcdf
module load hdf5

# Activer venv APRÈS les modules
source ~/venv_cgnet/bin/activate

echo "Which python after venv:"
which python

echo "Which pip after venv:"
which pip

python --version

cd /home/remilalonde/projects/def-sponsor00/remilalonde/2020/ClimateNet

python example.py

echo "Job finished"