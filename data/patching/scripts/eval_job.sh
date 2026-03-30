#!/bin/bash
#SBATCH --job-name=climatenet_eval
#SBATCH --account=def-sponsor00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1g.5gb:1
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

echo "========================================"
echo "Job ID     : $SLURM_JOB_ID"
echo "Nœud       : $SLURMD_NODENAME"
echo "Date début : $(date)"
echo "========================================"

REPO_DIR=$HOME/ClimateNet
DATA_SRC=/home/laurentfaucher/projects/def-sponsor00/shared_CN_B/climatenet_engineered
DATA_DIR=$SLURM_TMPDIR/climatenet

WEIGHTS=$HOME/ClimateNet/results/patches_run_1172/model/weights.pth
STRIDE=64   # 128 = pas d'overlap, 64 = 50% overlap, 32 = 75% overlap

# ── Copie des données ─────────────────────────────────────────────────────────
echo "[$(date)] Copie des données..."
cp -r $DATA_SRC $DATA_DIR

mkdir -p $DATA_DIR/val
for file in $DATA_DIR/train/*.nc; do
    year=$(basename "$file" | cut -d'-' -f2)
    if [[ $year -ge 2008 && $year -le 2010 ]]; then
        mv "$file" $DATA_DIR/val/
    fi
done
rm -f $DATA_DIR/val/data-2008-10-03-01-1_0.nc

echo "Val  : $(ls $DATA_DIR/val  | wc -l) fichiers"
echo "Test : $(ls $DATA_DIR/test | wc -l) fichiers"

# ── Virtualenv ────────────────────────────────────────────────────────────────
module --force purge
export PATH=/cvmfs/soft.computecanada.ca/easybuild/software/2023/x86-64-v3/Compiler/gcccore/python/3.11.5/bin:$PATH
export PIP_NO_CACHE_DIR=1

WH=/cvmfs/soft.computecanada.ca/custom/python/wheelhouse/gentoo2023/generic

virtualenv --no-download $SLURM_TMPDIR/venv
source $SLURM_TMPDIR/venv/bin/activate
pip install --upgrade pip --quiet

pip install $WH/torch-2.9.1+computecanada-cp311-cp311-linux_x86_64.whl
pip install $WH/numpy-1.26.4+computecanada-cp311-cp311-linux_x86_64.whl
pip install xarray "pandas<2" tqdm pyyaml h5py h5netcdf --quiet
pip install -e $REPO_DIR --quiet

echo "========================================"
python -c "import torch; print('torch:', torch.__version__, '| CUDA:', torch.cuda.is_available())"
echo "========================================"

# ── Évaluation ────────────────────────────────────────────────────────────────
python $REPO_DIR/eval_full_grid.py \
    -c       $REPO_DIR/config.json \
    -w       $WEIGHTS \
    -n       $DATA_DIR \
    --split  both \
    --stride $STRIDE \
    2>&1 | tee $HOME/ClimateNet/results/eval_stride${STRIDE}_$SLURM_JOB_ID.log

echo "========================================"
echo "[$(date)] Terminé."
