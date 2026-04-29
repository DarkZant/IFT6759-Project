#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --array=0-3  # Remove for a single job
#SBATCH --output=/project/def-sponsor00/etiennemb/logs/climatenet_%A_%a.txt  # %j for simple job id if no array
#SBATCH --export=NONE

# Initialisation
source /etc/profile

# Chargement des modules
module load python/3.10
module load openmpi/4.1.1
module load mpi4py/3.1.4

# Création et activation de l'environnement
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate

# Installation des paquets
pip install --no-index netCDF4 torch torchvision scipy xarray h5netcdf h5py

# Configuration pour fichiers HDF5
export HDF5_USE_FILE_LOCKING=FALSE

FEATURE_SETS=("cgnet" "all_engi" "non_engi" "cg_engi")

CURRENT_FEATURE=${FEATURE_SETS[$SLURM_ARRAY_TASK_ID]}

echo "This is Array Task $SLURM_ARRAY_TASK_ID. Training on feature set: $CURRENT_FEATURE"

# Lancement de l'entraînement
python -u train_attention_unet.py --features "$CURRENT_FEATURE" --epochs 25 --lr 1e-3

