#!/bin/bash
#SBATCH --account=def-sponsor00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --output=/project/def-sponsor00/medinammartin/logs/climatenet_%j.out
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
pip install --no-index netCDF4 torch torchvision scipy xarray h5netcdf h5py matplotlib

# Configuration pour fichiers HDF5
export HDF5_USE_FILE_LOCKING=FALSE

# Lancement de l'entraînement
python -u train_attention_unet.py

