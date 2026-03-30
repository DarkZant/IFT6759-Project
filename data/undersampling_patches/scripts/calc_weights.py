"""
calc_weights.py
───────────────
Calcule les class weights à partir des patches .npy générés par preprocess_patches.py.

Usage :
  python calc_weights.py --patch_dir /path/to/patches/train
"""

import numpy as np
import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--patch_dir', required=True, help='Dossier contenant les patches .npy')
args = parser.parse_args()

files = glob.glob(f'{args.patch_dir}/*.npy')
print(f'Patches trouvés : {len(files)}')

counts = np.zeros(3, dtype=np.int64)
for f in files:
    patch = np.load(f, allow_pickle=True).item()
    y = patch['y'].flatten()
    for cls in range(3):
        counts[cls] += np.sum(y == cls)

total = counts.sum()
freq  = counts / total

print(f'\nCounts : BG={counts[0]:,}  TC={counts[1]:,}  AR={counts[2]:,}')
print(f'Freq   : BG={freq[0]:.4f}  TC={freq[1]:.4f}  AR={freq[2]:.4f}')

# Inverse sqrt
w_sqrt = 1 / np.sqrt(freq)
w_sqrt /= w_sqrt.min()
print(f'\nInverse sqrt  : BG={w_sqrt[0]:.4f}  TC={w_sqrt[1]:.4f}  AR={w_sqrt[2]:.4f}')

# Median frequency
med = np.median(freq)
w_med = med / freq
print(f'Median freq   : BG={w_med[0]:.4f}  TC={w_med[1]:.4f}  AR={w_med[2]:.4f}')

# Output formaté pour --ce_weights
print(f'\n Pour --ce_weights (inverse sqrt)  : {w_sqrt[0]:.4f} {w_sqrt[1]:.4f} {w_sqrt[2]:.4f}')
print(f' Pour --ce_weights (median freq)   : {w_med[0]:.4f} {w_med[1]:.4f} {w_med[2]:.4f}')
