"""
preprocess_patches.py
─────────────────────
Extrait des patches 128×128 depuis les fichiers .nc de ClimateNet.
Garde uniquement les 4 variables (TMQ, U850, V850, PSL) sans normalisation.

Structure d'un patch sauvegardé (dict) :
  {
    'X': np.ndarray (4, 128, 128) float32  ← features brutes
    'y': np.ndarray     (128, 128) int64   ← labels 0/1/2
  }

Usage :
  python preprocess_patches.py \
      --data_dir  /path/to/climatenet/train \
      --out_dir   /path/to/patches/train \
      --patch_size 128 \
      --bg_ratio   0.30 \
      --seed       42
"""

import xarray as xr
import numpy as np
import argparse
import glob
import os
from pathlib import Path
from tqdm import tqdm

VARIABLES = ['TMQ', 'U850', 'V850', 'PSL']

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir',   required=True,  help='Dossier contenant les .nc')
parser.add_argument('--out_dir',    required=True,  help='Dossier de sortie pour les patches .npy')
parser.add_argument('--patch_size', type=int,   default=128)
parser.add_argument('--bg_ratio',   type=float, default=0.30, help='Fraction patches BG pur à garder (0.0-1.0)')
parser.add_argument('--seed',       type=int,   default=42)
args = parser.parse_args()

np.random.seed(args.seed)
P = args.patch_size
os.makedirs(args.out_dir, exist_ok=True)

# scan des fichiers
files = sorted(glob.glob(os.path.join(args.data_dir, '*.nc')))
print(f'Fichiers trouvés : {len(files)}')
print(f'Variables        : {VARIABLES}')
print(f'Patch size       : {P}×{P}')
print(f'BG ratio         : {args.bg_ratio*100:.0f}%')
print(f'Output dir       : {args.out_dir}')
print()

n_rows = 768  // P   # 6
n_cols = 1152 // P   # 9
print(f'Patches par fichier : {n_rows}×{n_cols} = {n_rows*n_cols}')
print()

total_tc         = 0
total_ar         = 0
total_bg_kept    = 0
total_bg_dropped = 0

for f in tqdm(files, desc='Extraction patches'):
    ds = xr.open_dataset(f)

    # Features : (4, 768, 1152) uniquement les 4 variables, pas de normalisation
    X = np.stack([ds[v].values[0] for v in VARIABLES], axis=0).astype(np.float32)

    # Labels : (768, 1152)
    y = ds['LABELS'].values.astype(np.int64)
    ds.close()

    stem = Path(f).stem  # ex: data-1996-06-09-01-1

    for i in range(n_rows):
        for j in range(n_cols):
            r0, r1 = i * P, (i + 1) * P
            c0, c1 = j * P, (j + 1) * P

            X_patch = X[:, r0:r1, c0:c1]   # (4, 128, 128)
            y_patch = y[r0:r1, c0:c1]       # (128, 128)

            has_tc = np.any(y_patch == 1)
            has_ar = np.any(y_patch == 2)

            if has_tc:
                label_type = 'tc'
                total_tc += 1
            elif has_ar:
                label_type = 'ar'
                total_ar += 1
            else:
                # Patch 100% BG → garder avec probabilité bg_ratio
                if np.random.random() < args.bg_ratio:
                    label_type = 'bg'
                    total_bg_kept += 1
                else:
                    total_bg_dropped += 1
                    continue

            patch_name = f'{stem}_r{i:02d}_c{j:02d}_{label_type}.npy'
            np.save(
                os.path.join(args.out_dir, patch_name),
                {'X': X_patch, 'y': y_patch}
            )

total_kept = total_tc + total_ar + total_bg_kept
total_all  = total_kept + total_bg_dropped

print()
print('=' * 50)
print('RÉSUMÉ')
print('=' * 50)
print(f'Total patches extraits    : {total_all:>8,}')
print(f'  → TC                    : {total_tc:>8,}  (100% gardés)')
print(f'  → AR                    : {total_ar:>8,}  (100% gardés)')
print(f'  → BG gardés ({args.bg_ratio*100:.0f}%)      : {total_bg_kept:>8,}')
print(f'  → BG supprimés          : {total_bg_dropped:>8,}')
print(f'Total patches sauvegardés : {total_kept:>8,}')
print(f'Shape par patch           : X=(4, {P}, {P})  y=({P}, {P})')
print(f'Taille estimée sur disque : ~{total_kept * P * P * 4 * 4 / 1e9:.2f} GB')
print(f'Output : {args.out_dir}')