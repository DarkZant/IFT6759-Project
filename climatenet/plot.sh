#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:10:00
#SBATCH --output=/project/def-sponsor00/remilalonde/logs/plot_%j.out

echo "Hostname:"
hostname

echo "Starting plotting job $SLURM_JOB_ID"

module --force purge
module load python/3.10

source ~/venv_clean/bin/activate

cd /home/remilalonde/projects/def-sponsor00/remilalonde/2020/ClimateNet

python plot_history.py

echo "Plot job finished"