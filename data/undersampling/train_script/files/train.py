from climatenet.utils.data import ClimateDatasetLabeled, ClimateDataset
from climatenet.models import CGNet
from climatenet.utils.utils import Config
from os import path
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config',  required=True, help='Path to config.json')
parser.add_argument('-d', '--data',    required=True, help='Path to climatenet data dir (contains train/, val/, test/)')
parser.add_argument('-o', '--output',  required=True, help='Path to save the trained model')
args = parser.parse_args()

print(f'Config   : {args.config}')
print(f'Data dir : {args.data}')
print(f'Output   : {args.output}')

# ── Chargement config & modèle ────────────────────────────────────────────────
config = Config(args.config)
model  = CGNet(config)

# ── Datasets ──────────────────────────────────────────────────────────────────
train_set = ClimateDatasetLabeled(path.join(args.data, 'train'), model.config)
val_set   = ClimateDatasetLabeled(path.join(args.data, 'val'),   model.config)
test_set  = ClimateDatasetLabeled(path.join(args.data, 'test'),  model.config)

print(f'Train : {len(train_set)} fichiers')
print(f'Val   : {len(val_set)} fichiers')
print(f'Test  : {len(test_set)} fichiers')

# ── Entraînement ──────────────────────────────────────────────────────────────
print('\n=== Entraînement ===')
model.train(train_set)
model.save_model(args.output)
print(f'Modèle sauvegardé dans : {args.output}')

# ── Évaluation ────────────────────────────────────────────────────────────────
print('\n=== Évaluation — Val set ===')
model.evaluate(val_set)

print('\n=== Évaluation — Test set ===')
model.evaluate(test_set)
