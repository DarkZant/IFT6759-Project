#!/bin/bash
#SBATCH --job-name=climatenet_top50
#SBATCH --account=def-sponsor00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --mem=16G
#SBATCH --time=04:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ── Infos ─────────────────────────────────────────────────────────────────────
echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Nœud       : $SLURMD_NODENAME"
echo "Date début : $(date)"
echo "Sélection  : top 64% (199 fichiers)"
echo "========================================"

# ── Chemins ───────────────────────────────────────────────────────────────────
REPO_DIR=$HOME/ClimateNet
DATA_SRC=/home/laurentfaucher/projects/def-sponsor00/shared_CN_B/climatenet_engineered
SELECTED_CSV=$REPO_DIR/selected_files_top64pct.csv

# ── Vérification du CSV ───────────────────────────────────────────────────────
if [ ! -f "$SELECTED_CSV" ]; then
    echo "❌ ERREUR : CSV introuvable à $SELECTED_CSV"
    exit 1
fi

N_SELECTED=$(tail -n +2 "$SELECTED_CSV" | wc -l)
echo "Fichiers sélectionnés dans le CSV : $N_SELECTED"

# ── Copie test complet ────────────────────────────────────────────────────────
echo "[$(date)] Copie du test set..."
mkdir -p $SLURM_TMPDIR/climatenet/test
cp $DATA_SRC/test/*.nc $SLURM_TMPDIR/climatenet/test/
echo "[$(date)] Test set : $(ls $SLURM_TMPDIR/climatenet/test | wc -l) fichiers"

# ── Copie sélective du train (top 50% par score pondéré) ─────────────────────
echo "[$(date)] Copie sélective du train (top 50%)..."
mkdir -p $SLURM_TMPDIR/climatenet/train

tail -n +2 "$SELECTED_CSV" | cut -d',' -f1 | while read filename; do
    src="$DATA_SRC/train/$filename"
    if [ -f "$src" ]; then
        cp "$src" $SLURM_TMPDIR/climatenet/train/
    else
        echo "  ⚠️  Fichier introuvable : $filename"
    fi
done

echo "[$(date)] Train copié : $(ls $SLURM_TMPDIR/climatenet/train | wc -l) fichiers"

# ── Suppression des fichiers corrompus ────────────────────────────────────────
echo "[$(date)] Suppression des fichiers corrompus..."
rm -f $SLURM_TMPDIR/climatenet/train/data-2002-12-27-01-1_4.nc
rm -f $SLURM_TMPDIR/climatenet/train/data-1997-10-12-01-1_0.nc
rm -f $SLURM_TMPDIR/climatenet/train/data-2001-10-29-01-1_3.nc
rm -f $SLURM_TMPDIR/climatenet/train/data-2000-04-17-01-1_5.nc
echo "[$(date)] Train après nettoyage : $(ls $SLURM_TMPDIR/climatenet/train | wc -l) fichiers"

# ── Val split (2008-2010) ─────────────────────────────────────────────────────
echo "[$(date)] Création du val set (années 2008-2010)..."
mkdir -p $SLURM_TMPDIR/climatenet/val

for file in $SLURM_TMPDIR/climatenet/train/*.nc; do
    year=$(basename "$file" | cut -d'-' -f2)
    if [[ $year -ge 2008 && $year -le 2010 ]]; then
        mv "$file" $SLURM_TMPDIR/climatenet/val/
    fi
done

rm -f $SLURM_TMPDIR/climatenet/val/data-2008-10-03-01-1_0.nc

echo "Val set   : $(ls $SLURM_TMPDIR/climatenet/val   | wc -l) fichiers"
echo "Train set : $(ls $SLURM_TMPDIR/climatenet/train | wc -l) fichiers"

# ── Virtualenv ────────────────────────────────────────────────────────────────
echo "[$(date)] Création du virtualenv..."
module --force purge
export PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.11.5/bin:$PATH
export PIP_NO_CACHE_DIR=1

WH=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --upgrade pip

pip install $WH/torch-2.9.1+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/torchvision-0.24.1+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/numpy-1.26.4+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/scipy-1.15.1+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/matplotlib-3.10.8+computecanada-cp311-cp311-linux_x86_64.whl

pip install xarray "pandas<2" tqdm pyyaml haversine Pillow requests \
            python-dateutil pytz six h5py h5netcdf

# Installation du package ClimateNet depuis le repo cloné
pip install -e $REPO_DIR

echo "========================================"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
python -c "import numpy; print('numpy:', numpy.__version__)"
python -c "import xarray; print('xarray:', xarray.__version__)"
python -c "from climatenet.models import CGNet; print('climatenet: OK')"
echo "========================================"

# ── Dossier de sortie ─────────────────────────────────────────────────────────
OUTPUT_DIR=$HOME/ClimateNet/results/top50_run_$SLURM_JOB_ID
mkdir -p $OUTPUT_DIR

echo "top50pct | alpha=12 beta=1 | $(date) | job=$SLURM_JOB_ID" > $OUTPUT_DIR/run_info.txt
echo "Train : $(ls $SLURM_TMPDIR/climatenet/train | wc -l) fichiers" >> $OUTPUT_DIR/run_info.txt
echo "Val   : $(ls $SLURM_TMPDIR/climatenet/val   | wc -l) fichiers" >> $OUTPUT_DIR/run_info.txt
echo "Test  : $(ls $SLURM_TMPDIR/climatenet/test  | wc -l) fichiers" >> $OUTPUT_DIR/run_info.txt

# ── Entraînement ──────────────────────────────────────────────────────────────
export PYTORCH_ALLOC_CONF=expandable_segments:True
echo "[$(date)] Démarrage entraînement — top 50% fichiers"
echo "========================================"

python $REPO_DIR/train.py \
    -c $REPO_DIR/config.json \
    -d $SLURM_TMPDIR/climatenet \
    -o $OUTPUT_DIR/model \
    2>&1 | tee $OUTPUT_DIR/train.log

echo "========================================"
echo "[$(date)] Terminé. Résultats dans : $OUTPUT_DIR"
