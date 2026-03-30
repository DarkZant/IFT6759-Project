#!/bin/bash
#SBATCH --job-name=climatenet_patches
#SBATCH --account=def-sponsor00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err


echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Nœud       : $SLURMD_NODENAME"
echo "Date début : $(date)"
echo "Strategy   : Patch-based 128x128, BG ratio 20%, ALL FEATURES"
echo "========================================"

REPO_DIR=$HOME/ClimateNet
DATA_SRC=/home/laurentfaucher/projects/def-sponsor00/shared_CN_B/climatenet_engineered
PATCH_DIR=$SLURM_TMPDIR/patches
DATA_DIR=$SLURM_TMPDIR/climatenet

echo "[$(date)] Copie des données..."
cp -r $DATA_SRC $DATA_DIR

rm -f $DATA_DIR/train/data-2002-12-27-01-1_4.nc
rm -f $DATA_DIR/train/data-1997-10-12-01-1_0.nc
rm -f $DATA_DIR/train/data-2001-10-29-01-1_3.nc
rm -f $DATA_DIR/train/data-2000-04-17-01-1_5.nc
echo "[$(date)] Copie terminée."

# Création de l'ensemble de validation
echo "[$(date)] Création du val set (années 2008-2010)..."
mkdir -p $DATA_DIR/val

for file in $DATA_DIR/train/*.nc; do
    year=$(basename "$file" | cut -d'-' -f2)
    if [[ $year -ge 2008 && $year -le 2010 ]]; then
        mv "$file" $DATA_DIR/val/
    fi
done

rm -f $DATA_DIR/val/data-2008-10-03-01-1_0.nc

echo "Val set   : $(ls $DATA_DIR/val   | wc -l) fichiers"
echo "Train set : $(ls $DATA_DIR/train | wc -l) fichiers"
echo "Test set  : $(ls $DATA_DIR/test  | wc -l) fichiers"

# Création de l'environement virtuel
echo "[$(date)] Création du virtualenv..."
module --force purge
export PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.11.5/bin:$PATH
export PIP_NO_CACHE_DIR=1

WH=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --upgrade pip --quiet

pip install $WH/torch-2.9.1+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/torchvision-0.24.1+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/numpy-1.26.4+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/scipy-1.15.1+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/matplotlib-3.10.8+computecanada-cp311-cp311-linux_x86_64.whl

pip install xarray "pandas<2" tqdm pyyaml haversine Pillow requests \
            python-dateutil pytz six h5py h5netcdf --quiet

pip install -e $REPO_DIR --quiet

echo "========================================"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "from climatenet.models import CGNet; print('climatenet: OK')"
echo "========================================"

# Preprocessing des patches
echo "[$(date)] Extraction des patches..."
mkdir -p $PATCH_DIR/train $PATCH_DIR/val $PATCH_DIR/test

python $REPO_DIR/preprocess_patches.py \
    --data_dir  $DATA_DIR/train \
    --out_dir   $PATCH_DIR/train \
    --patch_size 128 \
    --bg_ratio   0.20 \
    --seed       42

echo "[$(date)] Patches train : $(ls $PATCH_DIR/train | wc -l)"

python $REPO_DIR/preprocess_patches.py \
    --data_dir  $DATA_DIR/val \
    --out_dir   $PATCH_DIR/val \
    --patch_size 128 \
    --bg_ratio   1.00 \
    --seed       42

echo "[$(date)] Patches val : $(ls $PATCH_DIR/val | wc -l)"

# Pour le test on garde toutes les patches
python $REPO_DIR/preprocess_patches.py \
    --data_dir  $DATA_DIR/test \
    --out_dir   $PATCH_DIR/test \
    --patch_size 128 \
    --bg_ratio   1.00 \
    --seed       42

echo "[$(date)] Patches test : $(ls $PATCH_DIR/test | wc -l)"

OUTPUT_DIR=$HOME/ClimateNet/results/patches_run_$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR

# Entraînement
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[$(date)] Démarrage entraînement — patch-based"
echo "========================================"

python $REPO_DIR/train_patches_aug.py \
    -c $REPO_DIR/config.json \
    -d $PATCH_DIR \
    -o $OUTPUT_DIR/model \
    -n $DATA_DIR \
    2>&1 | tee $OUTPUT_DIR/train.log

echo "========================================"
echo "[$(date)] Terminé. Résultats dans : $OUTPUT_DIR"
