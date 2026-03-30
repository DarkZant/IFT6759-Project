"""
eval_full_grid.py
─────────────────
Évalue un modèle CGNet entraîné sur patches sur la grille complète 768×1152.
Supporte le sliding window avec overlap pour lisser les artefacts de bords.

Usage :
  # Sans overlap (stride = patch_size)
  python eval_full_grid.py -c config.json -w weights.pth -n nc_dir --split test

  # Avec overlap 50% (stride = patch_size / 2)
  python eval_full_grid.py -c config.json -w weights.pth -n nc_dir --split test --stride 64
"""

import argparse
import numpy as np
import torch
import glob
import xarray as xr
from tqdm import tqdm
from os import path

from climatenet.models import CGNetModule
from climatenet.utils.metrics import get_cm, get_iou_perClass
from climatenet.utils.utils import Config

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',     required=True, help='Chemin vers config.json')
parser.add_argument('-w', '--weights',    required=True, help='Chemin vers weights.pth')
parser.add_argument('-n', '--nc_dir',     required=True, help='Dossier contenant val/ et test/')
parser.add_argument('--split',            default='test', choices=['val', 'test', 'both'])
parser.add_argument('--patch_size',       type=int, default=128)
parser.add_argument('--stride',           type=int, default=None,
                    help='Stride du sliding window. Par défaut = patch_size (pas d overlap).'
                         ' Ex: 64 pour 50%% overlap avec patch_size=128')
args = parser.parse_args()

# Si stride non spécifié → pas d'overlap
if args.stride is None:
    args.stride = args.patch_size

P = args.patch_size
S = args.stride
overlap = P - S

print(f'Config      : {args.config}')
print(f'Weights     : {args.weights}')
print(f'NC dir      : {args.nc_dir}')
print(f'Split       : {args.split}')
print(f'Patch size  : {P}')
print(f'Stride      : {S}')
print(f'Overlap     : {overlap} pixels ({100*overlap/P:.0f}%)')

# ── Chargement config & modèle ────────────────────────────────────────────────
config  = Config(args.config)
network = CGNetModule(classes=3, channels=4).cuda()
network.load_state_dict(torch.load(args.weights))
network.eval()
print('✅ Poids chargés')

# ── Stats de normalisation ────────────────────────────────────────────────────
VARIABLES = ['TMQ', 'U850', 'V850', 'PSL']
means = torch.tensor(
    [config.fields[v]['mean'] for v in VARIABLES],
    dtype=torch.float32
).view(1, 4, 1, 1).cuda()
stds = torch.tensor(
    [config.fields[v]['std'] for v in VARIABLES],
    dtype=torch.float32
).view(1, 4, 1, 1).cuda()

# ── Fonction d'évaluation sliding window ──────────────────────────────────────
def evaluate_full_grid(nc_dir, split_name):
    files = sorted(glob.glob(path.join(nc_dir, '*.nc')))
    print(f'\n=== {split_name} — {len(files)} fichiers ===')

    aggregate_cm = np.zeros((3, 3))
    H, W = 768, 1152

    for f in tqdm(files, desc=split_name):
        ds = xr.open_dataset(f)
        X_full = np.stack([ds[v].values[0] for v in VARIABLES], axis=0).astype(np.float32)
        y_full = ds['LABELS'].values.astype(np.int64)  # (768, 1152)
        ds.close()

        # Accumulateurs pour le sliding window
        # pred_proba : somme des probabilités softmax sur les zones de chevauchement
        # count      : nombre de fois que chaque pixel a été prédit
        pred_proba = np.zeros((3, H, W), dtype=np.float32)
        count      = np.zeros((H, W),    dtype=np.float32)

        # Positions de départ du sliding window
        row_starts = list(range(0, H - P + 1, S))
        col_starts = list(range(0, W - P + 1, S))

        # S'assurer que le dernier patch couvre bien le bord droit/bas
        if row_starts[-1] + P < H:
            row_starts.append(H - P)
        if col_starts[-1] + P < W:
            col_starts.append(W - P)

        for r0 in row_starts:
            for c0 in col_starts:
                r1 = r0 + P
                c1 = c0 + P

                X_patch = torch.FloatTensor(X_full[:, r0:r1, c0:c1]) \
                               .unsqueeze(0).cuda()   # (1, 4, P, P)
                X_patch = (X_patch - means) / stds

                with torch.no_grad():
                    proba = torch.softmax(network(X_patch), 1)  # (1, 3, P, P)

                # Accumulation des probabilités
                pred_proba[:, r0:r1, c0:c1] += proba[0].cpu().numpy()
                count[r0:r1, c0:c1]         += 1.0

        # Moyenne des probabilités → classe finale
        pred_proba /= count[np.newaxis, :, :]           # (3, 768, 1152)
        pred_full   = np.argmax(pred_proba, axis=0)     # (768, 1152)

        aggregate_cm += get_cm(
            torch.tensor(pred_full),
            torch.tensor(y_full),
            3
        )

    print(f'Confusion matrix:\n{aggregate_cm}')
    ious = get_iou_perClass(aggregate_cm)
    print(f'IOUs: {ious}, mean: {ious.mean():.4f}')

    # Comparaison avec baseline
    baseline = {
        'TC': {'iou': 0.3486, 'precision': 0.4599, 'recall': 0.5903, 'spec': 0.9961},
        'AR': {'iou': 0.3848, 'precision': 0.4543, 'recall': 0.7157, 'spec': 0.9491},
    }
    class_names = ['Background', 'TC', 'AR']
    print()
    for cls_idx, cls_name in enumerate(['TC', 'AR']):
        i = cls_idx + 1
        TP = aggregate_cm[i, i]
        FP = aggregate_cm[:, i].sum() - TP
        FN = aggregate_cm[i, :].sum() - TP
        TN = aggregate_cm.sum() - TP - FP - FN
        iou  = TP / (TP + FP + FN + 1e-7)
        prec = TP / (TP + FP + 1e-7)
        rec  = TP / (TP + FN + 1e-7)
        spec = TN / (TN + FP + 1e-7)
        b = baseline[cls_name]
        print(f'--- {cls_name} ---')
        print(f'  IoU       : {iou:.4f}  (baseline {b["iou"]:.4f}  Δ{iou-b["iou"]:+.4f})')
        print(f'  Précision : {prec:.4f}  (baseline {b["precision"]:.4f}  Δ{prec-b["precision"]:+.4f})')
        print(f'  Rappel    : {rec:.4f}  (baseline {b["recall"]:.4f}  Δ{rec-b["recall"]:+.4f})')
        print(f'  Spéc.     : {spec:.4f}  (baseline {b["spec"]:.4f}  Δ{spec-b["spec"]:+.4f})')

    return ious

# ── Évaluation ────────────────────────────────────────────────────────────────
if args.split in ['val', 'both']:
    evaluate_full_grid(path.join(args.nc_dir, 'val'), 'Val')

if args.split in ['test', 'both']:
    evaluate_full_grid(path.join(args.nc_dir, 'test'), 'Test')
