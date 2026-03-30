"""
train_patches.py
────────────────
Script d'entraînement pour l'approche patch-based.
- Oversampling TC avec data augmentation (rotations + flips)
- Jaccard loss
- Monitoring val sur patches (rapide)
- Évaluation finale sur grille complète 768×1152 (comparable à la baseline)
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
import glob
import xarray as xr
import pathlib
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from os import path

from climatenet.models import CGNetModule
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from climatenet.utils.utils import Config
from patch_dataset_aug import PatchDataset

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',     required=True)
parser.add_argument('-d', '--data',       required=True,
                    help='Dossier patches (contient train/, val/)')
parser.add_argument('-n', '--nc_dir',     required=True,
                    help='Dossier .nc originaux (contient val/, test/)')
parser.add_argument('-o', '--output',     required=True)
parser.add_argument('--tc_oversample',    type=int, default=1,
                    help='Facteur oversampling TC (défaut=1, pas d oversampling)')
parser.add_argument('--ce_weights',       type=float, nargs=3, default=None,
                    metavar=('BG', 'TC', 'AR'),
                    help='Poids CrossEntropyLoss [BG TC AR]. '
                         'Si non spécifié, utilise la Jaccard loss.')
args = parser.parse_args()

print(f'Config        : {args.config}')
print(f'Data dir      : {args.data}')
print(f'NC dir        : {args.nc_dir}')
print(f'Output        : {args.output}')
print(f'TC oversample : x{args.tc_oversample}')

config  = Config(args.config)
network = CGNetModule(classes=3, channels=4).cuda()
optimizer = Adam(network.parameters(), lr=config.lr)

if args.ce_weights:
    class_weights = torch.tensor(args.ce_weights, dtype=torch.float32).cuda()
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    use_ce = True
else:
    use_ce = False

train_set = PatchDataset(
    path.join(args.data, 'train'), config,
    tc_oversample=args.tc_oversample
)
val_set = PatchDataset(
    path.join(args.data, 'val'), config,
    tc_oversample=1
)

print(f'Train : {len(train_set)} patches')
print(f'Val   : {len(val_set)} patches')

train_loader = DataLoader(
    train_set,
    batch_size=config.train_batch_size,
    shuffle=True,
    collate_fn=PatchDataset.collate,
    num_workers=4
)
val_loader = DataLoader(
    val_set,
    batch_size=config.pred_batch_size,
    shuffle=False,
    collate_fn=PatchDataset.collate,
    num_workers=4
)

VARIABLES = ['TMQ', 'U850', 'V850', 'PSL']
means = torch.tensor(
    [config.fields[v]['mean'] for v in VARIABLES],
    dtype=torch.float32
).view(1, 4, 1, 1).cuda()
stds = torch.tensor(
    [config.fields[v]['std'] for v in VARIABLES],
    dtype=torch.float32
).view(1, 4, 1, 1).cuda()

def evaluate_patches(loader, name='Eval'):
    network.eval()
    aggregate_cm = np.zeros((3, 3))
    for X, y in tqdm(loader, desc=name):
        X = X.cuda()
        y = y.cuda()
        with torch.no_grad():
            outputs = torch.softmax(network(X), 1)
        predictions = torch.max(outputs, 1)[1]
        aggregate_cm += get_cm(predictions, y, 3)
    ious = get_iou_perClass(aggregate_cm)
    print(f'IOUs [{name}]: {ious}, mean: {ious.mean():.4f}')
    return ious

def evaluate_full_grid(nc_dir, name='Eval grille complète', stride=64):
    network.eval()
    files = sorted(glob.glob(path.join(nc_dir, '*.nc')))
    print(f'\n=== {name} — {len(files)} fichiers (stride={stride}) ===')

    aggregate_cm = np.zeros((3, 3))
    P = 128
    H, W = 768, 1152

    row_starts = list(range(0, H - P + 1, stride))
    col_starts = list(range(0, W - P + 1, stride))
    if row_starts[-1] + P < H:
        row_starts.append(H - P)
    if col_starts[-1] + P < W:
        col_starts.append(W - P)

    for f in tqdm(files, desc=name):
        ds = xr.open_dataset(f)
        X_full = np.stack([ds[v].values[0] for v in VARIABLES], axis=0).astype(np.float32)
        y_full = ds['LABELS'].values.astype(np.int64)
        ds.close()

        pred_proba = np.zeros((3, H, W), dtype=np.float32)
        count      = np.zeros((H, W),    dtype=np.float32)

        for r0 in row_starts:
            for c0 in col_starts:
                r1, c1 = r0 + P, c0 + P
                X_patch = torch.FloatTensor(X_full[:, r0:r1, c0:c1]) \
                               .unsqueeze(0).cuda()
                X_patch = (X_patch - means) / stds
                with torch.no_grad():
                    proba = torch.softmax(network(X_patch), 1)
                pred_proba[:, r0:r1, c0:c1] += proba[0].cpu().numpy()
                count[r0:r1, c0:c1]         += 1.0

        pred_proba /= count[np.newaxis, :, :]
        pred_full   = np.argmax(pred_proba, axis=0)
        aggregate_cm += get_cm(
            torch.tensor(pred_full),
            torch.tensor(y_full),
            3
        )

    print(f'Confusion matrix:\n{aggregate_cm}')
    ious = get_iou_perClass(aggregate_cm)
    print(f'IOUs: {ious}, mean: {ious.mean():.4f}')
    return ious

print(f'\n=== Entraînement ({config.epochs} epochs) ===')
for epoch in range(config.epochs):
    network.train()
    epoch_loader = tqdm(train_loader, desc=f'Epoch {epoch+1}')
    total_loss = 0

    for X, y in epoch_loader:
        X = X.cuda()
        y = y.cuda()

        if use_ce:
            outputs = network(X)              # logits (B, 3, H, W)
            loss = criterion(outputs, y)
        else:
            outputs = torch.softmax(network(X), 1)
            loss = jaccard_loss(outputs, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        epoch_loader.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} — avg loss: {avg_loss:.4f}')

    evaluate_patches(val_loader, name=f'Val patches epoch {epoch+1}')

pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
torch.save(network.state_dict(), path.join(args.output, 'weights.pth'))
print(f'\nModèle sauvegardé dans : {args.output}')

print('\n=== Évaluation finale — Val set (grille complète) ===')
evaluate_full_grid(path.join(args.nc_dir, 'val'), name='Val', stride=64)

print('\n=== Évaluation finale — Test set (grille complète) ===')
evaluate_full_grid(path.join(args.nc_dir, 'test'), name='Test', stride=64)
