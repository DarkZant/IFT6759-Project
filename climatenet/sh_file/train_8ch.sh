#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --job-name=cgnet_8ch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=/home/remilalonde/Project/IFT6759-Project/ClimateNet/logs/cgnet_8ch_%j.out
#SBATCH --error=/home/remilalonde/Project/IFT6759-Project/ClimateNet/logs/cgnet_8ch_%j.err

echo "Starting CGNet 8-channel job $SLURM_JOB_ID"

module --force purge
module load StdEnv/2023
module load gcc/12.3 openmpi/4.1.5 mpi4py/3.1.6
module load python/3.10.13

source /home/remilalonde/Project/IFT6759-Project/.venv/bin/activate

cd /home/remilalonde/Project/IFT6759-Project/ClimateNet

python example.py --config /home/remilalonde/Project/IFT6759-Project/climatenet/config_files/config_8ch.json --save_preds --output_dir /home/remilalonde/Project/IFT6759-Project/ClimateNet/outputs/cgnet_8ch

echo "Job finished"
