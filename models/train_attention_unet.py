import os
import argparse

import numpy as np

import torch
from torch import Tensor
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

import xarray as xr
import matplotlib.pyplot as plt
from netCDF4 import Dataset as NetCDFDataset

from typing import Tuple, List

from attention_unet import AttentionUNet

class FeatureSets:
    """Defines different feature sets for training and evaluation."""

    CGNet = "cgnet"
    EngiCG = "cg_engi"
    NonEngiKept = "non_engi"
    AllEngi = "all_engi"
    
    Engineered = "engi"
    Kept = "kept"
    Other = "other"
    All = "all"

    CGNetFeatures = ['TMQ', 'U850', 'V850', 'PSL']
    EngineeredFeatures = ['WS850', 'WSBOT', 'VRT850', 'VRTBOT']
    KeptFeatures = ['UBOT', 'VBOT', 'PS', 'PRECT']
    OtherFeatures = ['TS', 'TREFHT', 'Z1000', 'Z200', 'ZBOT', 'QREFHT', 'T200', 'T500']
    
    def __init__(self):
        self.feature_groups = {
            self.CGNet: self.CGNetFeatures,
            self.Engineered: self.EngineeredFeatures,
            self.Kept: self.KeptFeatures,
            self.Other: self.OtherFeatures,
            self.EngiCG: self.CGNetFeatures + self.EngineeredFeatures,
            self.NonEngiKept: self.CGNetFeatures + self.KeptFeatures,
            self.AllEngi: self.CGNetFeatures + self.EngineeredFeatures + self.KeptFeatures,
            self.All: self.CGNetFeatures + self.EngineeredFeatures + self.KeptFeatures + self.OtherFeatures
        }


    def get_features(self, set_name: str) -> List[str]:
        return self.feature_groups.get(set_name, self.All)
    

    def get_all_set_names(self) -> List[str]:
        return list(self.feature_groups.keys())
        

class ClimateNetDataset(Dataset):
    """Custom Dataset for loading ClimateNet data from NetCDF files."""

    def __init__(self, data_dir, feature_set, normalize=True):
        self.data_dir = data_dir
        self.feature_set = feature_set
        self.normalize = normalize

        # Corrupted files on dataset
        self.corrupted_files = {
            "data-2002-12-27-01-1_4.nc",
            "data-1997-10-12-01-1_0.nc",
            "data-2001-10-29-01-1_3.nc",
            "data-2000-04-17-01-1_5.nc",
            "data-2008-10-03-01-1_0.nc"
        }

        if os.path.exists(data_dir):
            all_files = os.listdir(data_dir)
            self.valid_files = [f for f in all_files if f.endswith('.nc') and f not in self.corrupted_files]
        else:
            self.valid_files = []
            raise FileNotFoundError(f"Directory {data_dir} not found. Please verify the path.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor, str]:
        """Returns (features, labels, timestamp) for the given index."""
        filename = self.valid_files[idx]
        file_path = os.path.join(self.data_dir, filename)
        target_shape = (768, 1152)

        try:
            with NetCDFDataset(file_path, 'r') as nc:
                labels = nc.variables['LABELS'][:].squeeze().astype(np.int64)

                processed_features = []
                for k in self.feature_set:
                    if k in nc.variables:
                        var_data = nc.variables[k][:].squeeze()
                        if len(var_data.shape) == 2:
                            if var_data.shape != target_shape:
                                var_data = var_data[:target_shape[0], :target_shape[1]]
                            processed_features.append(var_data)

                data = np.stack(processed_features, axis=0)

        except Exception as e:
            print(f"\nWARNING: Skipping corrupted file {file_path}. Error: {e}")
            # Recursively call __getitem__ with a random different index
            import random
            return self.__getitem__(random.randint(0, len(self.valid_files) - 1))

        # Standardize
        if self.normalize:
            data = (data - data.mean(axis=(1, 2), keepdims=True)) / (data.std(axis=(1, 2), keepdims=True) + 1e-7)

        timestamp_str = self._parse_timestamp(filename)

        return torch.from_numpy(data).float(), torch.from_numpy(labels), timestamp_str
    

    def _parse_timestamp(self, filename: str) -> str:
        """
        Extracts the date and calculates the hour from a ClimateNet filename.
        Example filename: data-2002-12-27-01-1_4.nc
        """
        # Split the string by dashes
        parts = filename.split('-') 
        
        year = parts[1]   # '2002'
        month = parts[2]  # '12'
        day = parts[3]    # '27'
        
        # parts[5] looks like '1_4.nc'. We split by '_' to isolate the timestep '1'
        timestep_str = parts[5].split('_')[0] 
        timestep = int(timestep_str)
        
        # Map timestep (1-8) to hour (0, 3, 6, 9, 12, 15, 18, 21)
        hour = (timestep - 1) * 3
        
        # Return a nicely formatted string
        return f"{year}-{month}-{day} {hour:02d}:00 UTC"
    

def evaluate(model, loader, device, num_classes=3):
    """Evaluates the model over the entire dataloader and computes the metrics."""
    model.eval()

    # Initialize metric
    total_TP = {c: 0 for c in range(num_classes)}
    total_FP = {c: 0 for c in range(num_classes)}
    total_FN = {c: 0 for c in range(num_classes)}
    total_TN = {c: 0 for c in range(num_classes)}

    with torch.no_grad():
        for images, targets, _ in loader:
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
            'IoU': TP / max((TP + FP + FN), 1),
            'Recall': TP / max((TP + FN), 1),
            'Precision': TP / max((TP + FP), 1),
            'Specificity': TN / max((TN + FP), 1)
        }
    return metrics


class DiceLoss(nn.Module):
    """Calculates Multiclass Dice Loss directly optimizing for IoU"""

    def __init__(self, smooth=1e-6):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, true):
        probs = F.softmax(logits, dim=1)
        # Convert true labels to one-hot encoding [Batch, Classes, Height, Width]
        true_1hot = F.one_hot(true, num_classes=probs.shape[1]).permute(0, 3, 1, 2).float()

        # Calculate intersection and cardinality across spatial and batch dimensions
        dims = (0, 2, 3)
        intersection = torch.sum(probs * true_1hot, dims)
        cardinality = torch.sum(probs + true_1hot, dims)

        dice = (2. * intersection + self.smooth) / (cardinality + self.smooth)
        return 1.0 - dice.mean()


def train(features: str, num_epochs: int, lr: float):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.abspath(os.path.join(current_dir, "../shared_CN_B/climatenet_engineered/train"))
    test_dir = os.path.abspath(os.path.join(current_dir, "../shared_CN_B/climatenet_engineered/test"))

    # Create Datasets and DataLoaders
    feature_sets = FeatureSets()
    feature_set = feature_sets.get_features(features)
    train_dataset = ClimateNetDataset(train_dir, feature_set)
    test_dataset = ClimateNetDataset(test_dir, feature_set)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    # Initialize model
    model = AttentionUNet(in_channels=len(feature_set), out_channels=3)

    if torch.cuda.device_count() > 1:
        print(f"Distributing model on {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Combined Loss setup
    weights = torch.tensor([0.5, 5.0, 2.0]).to(device)
    ce_loss_fn = nn.CrossEntropyLoss(weight=weights)
    dice_loss_fn = DiceLoss()

    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Gradient Accumulation setup: Effective batch size = batch_size(2) * accumulation_steps(4) = 8
    accumulation_steps = 4

    # Dictionaries to track history for ALL plots
    history = {
        'train_loss': [],
        'train_tc_iou': [], 'train_ar_iou': [], 'test_tc_iou': [], 'test_ar_iou': [],
        'train_tc_recall': [], 'train_ar_recall': [], 'test_tc_recall': [], 'test_ar_recall': [],
        'train_tc_precision': [], 'train_ar_precision': [], 'test_tc_precision': [], 'test_ar_precision': [],
        'train_tc_specificity': [], 'train_ar_specificity': [], 'test_tc_specificity': [], 'test_ar_specificity': []
    }

    print(f"Starting training on device: {device}")
    print(f"Total Train samples: {len(train_dataset)}")
    print(f"Total Test samples: {len(test_dataset)}")
    print(f'Training with feature set "{features}" for {num_epochs} epochs with learning rate {lr}')

    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        optimizer.zero_grad()  # Zero gradients at the start of the epoch

        for batch_idx, (images, labels, _) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)

            # Combine Cross-Entropy and Dice Loss
            loss_ce = ce_loss_fn(outputs, labels)
            loss_dice = dice_loss_fn(outputs, labels)
            loss = loss_ce + loss_dice

            # Normalize loss for accumulation
            loss = loss / accumulation_steps
            loss.backward()

            # Only update weights after accumulating gradients for 'accumulation_steps' batches
            if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                optimizer.step()
                optimizer.zero_grad()

            epoch_loss += (loss.item() * accumulation_steps)  # Keep true loss scale for logging

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {(loss.item() * accumulation_steps):.4f}")

        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)

        print(
            f"\n--- Epoch {epoch + 1}/{num_epochs} Complete | Avg Train Loss: {avg_train_loss:.4f} ---"
        )
        print("Evaluating model...")
        # Evaluate on the entire Train and Test sets
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        # Log History for ALL Metrics
        history['train_tc_iou'].append(train_metrics[1]['IoU'])
        history['train_ar_iou'].append(train_metrics[2]['IoU'])
        history['test_tc_iou'].append(test_metrics[1]['IoU'])
        history['test_ar_iou'].append(test_metrics[2]['IoU'])

        history['train_tc_recall'].append(train_metrics[1]['Recall'])
        history['train_ar_recall'].append(train_metrics[2]['Recall'])
        history['test_tc_recall'].append(test_metrics[1]['Recall'])
        history['test_ar_recall'].append(test_metrics[2]['Recall'])

        history['train_tc_precision'].append(train_metrics[1]['Precision'])
        history['train_ar_precision'].append(train_metrics[2]['Precision'])
        history['test_tc_precision'].append(test_metrics[1]['Precision'])
        history['test_ar_precision'].append(test_metrics[2]['Precision'])

        history['train_tc_specificity'].append(train_metrics[1]['Specificity'])
        history['train_ar_specificity'].append(train_metrics[2]['Specificity'])
        history['test_tc_specificity'].append(test_metrics[1]['Specificity'])
        history['test_ar_specificity'].append(test_metrics[2]['Specificity'])

        print(" [TRAIN METRICS]")
        print(
            f"  TC -> IoU: {train_metrics[1]['IoU']:.4f} | Recall: {train_metrics[1]['Recall']:.4f} | Precision: {train_metrics[1]['Precision']:.4f} | Specificity: {train_metrics[1]['Specificity']:.4f}")
        print(
            f"  AR -> IoU: {train_metrics[2]['IoU']:.4f} | Recall: {train_metrics[2]['Recall']:.4f} | Precision: {train_metrics[2]['Precision']:.4f} | Specificity: {train_metrics[2]['Specificity']:.4f}")

        print(" [TEST METRICS]")
        print(
            f"  TC -> IoU: {test_metrics[1]['IoU']:.4f} | Recall: {test_metrics[1]['Recall']:.4f} | Precision: {test_metrics[1]['Precision']:.4f} | Specificity: {test_metrics[1]['Specificity']:.4f}")
        print(
            f"  AR -> IoU: {test_metrics[2]['IoU']:.4f} | Recall: {test_metrics[2]['Recall']:.4f} | Precision: {test_metrics[2]['Precision']:.4f} | Specificity: {test_metrics[2]['Specificity']:.4f}\n")
        
        
    # --- SAVE FINAL MODEL ---
    print("\n" + "=" * 50)
    print("SAVING FINAL MODEL")
    print("=" * 50)
    torch.save(model.state_dict(), "final_attention_unet.pth")
    print("Model saved successfully as 'final_attention_unet.pth'.")

    # --- GRAPHS ---
    print("\nGenerating training plots...")
    epochs_range = range(1, len(history['train_loss']) + 1)

    os.makedirs('plots', exist_ok=True)

    # Loss Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_loss'], label='Train Loss', color='red', marker='o')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'plot_loss.png'))
    plt.close()

    # IoU Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_tc_iou'], label='Train TC IoU', linestyle='--', color='blue')
    plt.plot(epochs_range, history['train_ar_iou'], label='Train AR IoU', linestyle='--', color='green')
    plt.plot(epochs_range, history['test_tc_iou'], label='Test TC IoU', marker='o', color='blue')
    plt.plot(epochs_range, history['test_ar_iou'], label='Test AR IoU', marker='o', color='green')
    plt.title('Intersection over Union (IoU)')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'plot_iou.png'))
    plt.close()

    # Recall Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_tc_recall'], label='Train TC Recall', linestyle='--', color='blue')
    plt.plot(epochs_range, history['train_ar_recall'], label='Train AR Recall', linestyle='--', color='green')
    plt.plot(epochs_range, history['test_tc_recall'], label='Test TC Recall', marker='o', color='blue')
    plt.plot(epochs_range, history['test_ar_recall'], label='Test AR Recall', marker='o', color='green')
    plt.title('Recall')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'plot_recall.png'))
    plt.close()

    # Precision Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_tc_precision'], label='Train TC Precision', linestyle='--', color='blue')
    plt.plot(epochs_range, history['train_ar_precision'], label='Train AR Precision', linestyle='--', color='green')
    plt.plot(epochs_range, history['test_tc_precision'], label='Test TC Precision', marker='o', color='blue')
    plt.plot(epochs_range, history['test_ar_precision'], label='Test AR Precision', marker='o', color='green')
    plt.title('Precision')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'plot_precision.png'))
    plt.close()

    # Specificity Plot
    plt.figure(figsize=(8, 6))
    plt.plot(epochs_range, history['train_tc_specificity'], label='Train TC Specificity', linestyle='--', color='blue')
    plt.plot(epochs_range, history['train_ar_specificity'], label='Train AR Specificity', linestyle='--', color='green')
    plt.plot(epochs_range, history['test_tc_specificity'], label='Test TC Specificity', marker='o', color='blue')
    plt.plot(epochs_range, history['test_ar_specificity'], label='Test AR Specificity', marker='o', color='green')
    plt.title('Specificity')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join('plots', 'plot_specificity.png'))
    plt.close()

    print("\nTraining plots saved at /plots")
        

    model_save_path = "/project/def-sponsor00/etiennemb/models"
    torch.save(model.state_dict(), os.path.join(model_save_path, f"att_unet_{num_epochs}epo_feat_{features}.pth"))


def parse_args():
    parser = argparse.ArgumentParser(description='Attention UNet Training Script for ClimateNet Segmentation')
    parser.add_argument('--features', type=str, default="all", help='Features to use for training', choices=FeatureSets().get_all_set_names())
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs for training (default: %(default)s).')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: %(default)s).')
    return parser.parse_args()


def main():
    args = parse_args()
    train(args.features, args.epochs, args.lr)


if __name__ == "__main__":
    main()