"""
train_patches.py
────────────────
Script d'entraînement pour l'approche patch-based.
Utilise PatchDataset au lieu de ClimateDatasetLabeled.
"""

import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm
from os import path

from climatenet.modules import CGNetModule
from climatenet.utils.losses import jaccard_loss
from climatenet.utils.metrics import get_cm, get_iou_perClass
from climatenet.utils.utils import Config
from patch_dataset import PatchDataset

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',  required=True)
parser.add_argument('-d', '--data',    required=True, help='Dossier patches (contient train/, val/, test/)')
parser.add_argument('-o', '--output',  required=True)
args = parser.parse_args()

print(f'Config   : {args.config}')
print(f'Data dir : {args.data}')
print(f'Output   : {args.output}')

# ── Config & modèle ───────────────────────────────────────────────────────────
config = Config(args.config)

# Le modèle attend N_CHANNELS en entrée — on utilise toutes les 16 variables
N_CHANNELS = 4
network = CGNetModule(classes=3, channels=N_CHANNELS).cuda()
optimizer = Adam(network.parameters(), lr=config.lr)

# ── Datasets ──────────────────────────────────────────────────────────────────
train_set = PatchDataset(path.join(args.data, "train"), config)
val_set   = PatchDataset(path.join(args.data, 'val'), config)
test_set  = PatchDataset(path.join(args.data, 'test'), config)

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
test_loader = DataLoader(
    test_set,
    batch_size=config.pred_batch_size,
    shuffle=False,
    collate_fn=PatchDataset.collate,
    num_workers=4
)

# ── Entraînement ──────────────────────────────────────────────────────────────
def evaluate(loader, name='Eval'):
    network.eval()
    aggregate_cm = np.zeros((3, 3))
    for X, y in tqdm(loader, desc=name):
        X = X.cuda()
        y = y.cuda()
        with torch.no_grad():
            outputs = torch.softmax(network(X), 1)
        predictions = torch.max(outputs, 1)[1]
        aggregate_cm += get_cm(predictions, y, 3)

    print(f'Evaluation stats ({name}):')
    print(aggregate_cm)
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

        outputs = torch.softmax(network(X), 1)
        loss = jaccard_loss(outputs, y)

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        epoch_loader.set_postfix({'loss': f'{loss.item():.4f}'})

    avg_loss = total_loss / len(train_loader)
    print(f'Epoch {epoch+1} — avg loss: {avg_loss:.4f}')

    # Évaluation sur val à chaque epoch
    val_ious = evaluate(val_loader, name=f'Val epoch {epoch+1}')

# ── Sauvegarde ────────────────────────────────────────────────────────────────
import pathlib
pathlib.Path(args.output).mkdir(parents=True, exist_ok=True)
torch.save(network.state_dict(), path.join(args.output, 'weights.pth'))
print(f'\nModèle sauvegardé dans : {args.output}')

# ── Évaluation finale ─────────────────────────────────────────────────────────
print('\n=== Évaluation finale — Val set ===')
evaluate(val_loader, name='Val')

print('\n=== Évaluation finale — Test set ===')
evaluate(test_loader, name='Test')
