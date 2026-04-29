import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler
import xarray as xr
from conv_lstm import ConvLSTMUNet, ConvLSTMUNetFullLSTM, ConvLSTMPure, ConvLSTMUNetGC


# Arguments
def parse_args():
    parser = argparse.ArgumentParser(description="Entraînement ConvLSTM sur ClimateNet")
    parser.add_argument("--train_dir", type=str, required=True, help="Chemin vers les données d'entraînement")
    parser.add_argument("--test_dir", type=str, required=True, help="Chemin vers les données de test")
    parser.add_argument("--checkpoint_dir", type=str, required=True, help="Dossier de sauvegarde des checkpoints")
    parser.add_argument("--output_dir", type=str, required=True, help="Dossier de sauvegarde des résultats")
    parser.add_argument("--seq_len", type=int, default=2, help="Longueur des séquences temporelles (défaut: 2)")
    parser.add_argument("--epochs", type=int, default=15, help="Nombre d'époques (défaut: 15)")
    parser.add_argument("--batch_size", type=int, default=1, help="Taille des batches (défaut: 1)")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate (défaut: 0.001)")
    return parser.parse_args()


# Dataset
class ClimateNetSequenceDataset(Dataset):
    def __init__(self, data_dir, seq_len=2, variables=None):
        self.data_dir = data_dir
        self.seq_len = seq_len
        self.variables = variables or ['TMQ', 'U850', 'V850', 'PSL']
        # self.variables = variables or ['TMQ', 'WS850', 'VRT850', 'PSL', 'PRECT']
        # self.variables = variables or ['TMQ', 'U850', 'V850', 'UBOT', 'VBOT', 'QREFHT', 'PS', 'PSL', 'T200', 'T500', 'PRECT', 'TS', 'TREFHT', 'Z1000', 'Z200', 'ZBOT']

        # Fichiers corrompus à exclure
        corrupted = {
            "data-2002-12-27-01-1_4.nc",
            "data-1997-10-12-01-1_0.nc",
            "data-2001-10-29-01-1_3.nc"
        }

        if not os.path.exists(data_dir):
            raise FileNotFoundError(f"Directory {data_dir} not found. Please verify the path.")

        all_files = sorted([
            f for f in os.listdir(data_dir)
            if f.endswith('.nc') and f not in corrupted
        ])
        self.valid_files = all_files
        self.sequences = list(range(len(all_files) - seq_len + 1))

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        start = self.sequences[idx]
        frames = []
        labels = None

        for t in range(self.seq_len):
            path = os.path.join(self.data_dir, self.valid_files[start + t])
            with xr.open_dataset(path, engine='h5netcdf') as ds:
                # Label du dernier timestep de la séquence seulement
                if t == self.seq_len - 1:
                    labels = torch.from_numpy(
                        ds['LABELS'].squeeze().values.astype(np.int64)
                    )
                frame = ds[self.variables].to_array().squeeze().values
                frame = (frame - frame.mean(axis=(1, 2), keepdims=True)) / \
                        (frame.std(axis=(1, 2), keepdims=True) + 1e-7)
                frames.append(torch.from_numpy(frame).float())

        # x shape: (T, C, H, W)
        x = torch.stack(frames, dim=0)
        return x, labels
    

# Evaluation
def evaluate(model, loader, device, num_classes=3):
    model.eval()

    total_TP = {c: 0 for c in range(num_classes)}
    total_FP = {c: 0 for c in range(num_classes)}
    total_FN = {c: 0 for c in range(num_classes)}
    total_TN = {c: 0 for c in range(num_classes)}

    with torch.no_grad(): # on n'a pas besoin de calculer les gradients pour l'évaluation
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device).view(-1)

            outputs = model(images)
            preds = torch.argmax(outputs, dim=1).view(-1)

            for cls in range(num_classes):
                total_TP[cls] += ((preds == cls) & (targets == cls)).sum().item()
                total_FP[cls] += ((preds == cls) & (targets != cls)).sum().item()
                total_FN[cls] += ((preds != cls) & (targets == cls)).sum().item()
                total_TN[cls] += ((preds != cls) & (targets != cls)).sum().item()

    metrics = {}
    for cls in range(num_classes):
        TP = total_TP[cls]
        FP = total_FP[cls]
        FN = total_FN[cls]
        TN = total_TN[cls]

        metrics[cls] = {
            'IoU': TP / max((TP + FP + FN), 1), # éviter division par zéro
            'Recall': TP / max((TP + FN), 1),
            'Precision': TP / max((TP + FP), 1),
            'Specificity': TN / max((TN + FP), 1)
        }
    return metrics


# Entraînement
def train():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.output_dir, exist_ok=True)

    train_dataset = ClimateNetSequenceDataset(args.train_dir, seq_len=args.seq_len)
    test_dataset  = ClimateNetSequenceDataset(args.test_dir, seq_len=args.seq_len)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    # Pour expérience 14
    model = ConvLSTMUNetGC(in_channels=16, out_channels=3).to(device)

    # Mêmes poids que l'Attention UNet pour comparaison équitable
    weights = torch.tensor([0.5, 5.0, 2.0]).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # Mixed precision : réduit l'usage mémoire GPU d'environ 50%
    scaler = GradScaler()

    print(f"Démarrage de l'entraînement sur : {device}")
    print(f"Séquences train : {len(train_dataset)} | Séquences test : {len(test_dataset)}")
    print(f"Longueur de séquence : {args.seq_len} | Variables : TMQ, U850, V850, PSL")
    print(f"Époques : {args.epochs} | Batch size : {args.batch_size} | LR : {args.lr}\n")

    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx+1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / len(train_loader)
        print(f"\n--- Epoch {epoch+1}/{args.epochs} | Avg Loss: {avg_loss:.4f} ---")
    
        # Évaluation pour train et test
        train_metrics = evaluate(model, train_loader, device)
        test_metrics  = evaluate(model, test_loader,  device)

        for split, m in [("TRAIN", train_metrics), ("TEST", test_metrics)]:
            print(f"  [{split} METRICS]")
            print(f"  TC -> IoU: {m[1]['IoU']:.4f} | Recall: {m[1]['Recall']:.4f} | Precision: {m[1]['Precision']:.4f} | Specificity: {m[1]['Specificity']:.4f}")
            print(f"  AR -> IoU: {m[2]['IoU']:.4f} | Recall: {m[2]['Recall']:.4f} | Precision: {m[2]['Precision']:.4f} | Specificity: {m[2]['Specificity']:.4f}")

        # Sauvegarde checkpoint
        ckpt_path = os.path.join(args.checkpoint_dir, f"convlstm_epoch_{epoch+1}.pt")
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': avg_loss,
        }, ckpt_path)
        print(f"  Checkpoint sauvegardé : {ckpt_path}\n")


if __name__ == "__main__":
    train()