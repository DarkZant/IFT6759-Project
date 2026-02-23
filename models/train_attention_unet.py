import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import xarray as xr
import numpy as np
from attention_unet import AttentionUNet

class ClimateNetDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.corrupted_files = {
            "data-2002-12-27-01-1_4.nc",
            "data-1997-10-12-01-1_0.nc",
            "data-2001-10-29-01-1_3.nc"
        }

        if os.path.exists(data_dir):
            all_files = os.listdir(data_dir)
            self.valid_files = [f for f in all_files if f.endswith('.nc') and f not in self.corrupted_files]
        else:
            raise FileNotFoundError(f"Directory {data_dir} not found. Please verify the path.")

    def __len__(self):
        return len(self.valid_files)

    def __getitem__(self, idx):
        file_path = os.path.join(self.data_dir, self.valid_files[idx])

        with xr.open_dataset(file_path, engine='h5netcdf') as ds:
            labels = ds['LABELS'].squeeze().values.astype(np.int64)
            ds_features = ds.drop_vars('LABELS')
            data = ds_features.to_array().squeeze().values

        data = (data - data.mean(axis=(1, 2), keepdims=True)) / (data.std(axis=(1, 2), keepdims=True) + 1e-7)

        return torch.from_numpy(data).float(), torch.from_numpy(labels)


def evaluate(model, loader, device, num_classes=3):
    """Evaluates the model over the entire dataloader and computes the metrics."""
    model.eval()

    # Initialize metric counters for all classes
    total_TP = {c: 0 for c in range(num_classes)}
    total_FP = {c: 0 for c in range(num_classes)}
    total_FN = {c: 0 for c in range(num_classes)}
    total_TN = {c: 0 for c in range(num_classes)}

    with torch.no_grad():
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
            'IoU': TP / max((TP + FP + FN), 1),
            'Recall': TP / max((TP + FN), 1),
            'Precision': TP / max((TP + FP), 1),
            'Specificity': TN / max((TN + FP), 1)
        }
    return metrics


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_dir = os.path.abspath(os.path.join(current_dir, "../shared_CN_B/climatenet/train"))
    test_dir = os.path.abspath(os.path.join(current_dir, "../shared_CN_B/climatenet/test"))

    # Create Datasets and DataLoaders
    train_dataset = ClimateNetDataset(train_dir)
    test_dataset = ClimateNetDataset(test_dir)

    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=2, shuffle=False, num_workers=0)

    model = AttentionUNet(in_channels=16, out_channels=3).to(device)

    weights = torch.tensor([0.5, 5.0, 2.0]).to(device) # [0.1, 10.0, 3.0]
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Starting training on device: {device}")
    print(f"Total Train samples: {len(train_dataset)}")
    print(f"Total Test samples: {len(test_dataset)}")

    num_epochs = 15
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0

        for batch_idx, (images, labels) in enumerate(train_loader):
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

            if (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        print(
            f"\n--- Epoch {epoch + 1}/{num_epochs} Complete | Avg Train Loss: {epoch_loss / len(train_loader):.4f} ---")

        # Evaluate on the entire Train and Test sets
        train_metrics = evaluate(model, train_loader, device)
        test_metrics = evaluate(model, test_loader, device)

        print("  [TRAIN METRICS]")
        print(
            f"  TC -> IoU: {train_metrics[1]['IoU']:.4f} | Recall: {train_metrics[1]['Recall']:.4f} | Precision: {train_metrics[1]['Precision']:.4f} | Specificity: {train_metrics[1]['Specificity']:.4f}")
        print(
            f"  AR -> IoU: {train_metrics[2]['IoU']:.4f} | Recall: {train_metrics[2]['Recall']:.4f} | Precision: {train_metrics[2]['Precision']:.4f} | Specificity: {train_metrics[2]['Specificity']:.4f}")

        print("  [TEST METRICS]")
        print(
            f"  TC -> IoU: {test_metrics[1]['IoU']:.4f} | Recall: {test_metrics[1]['Recall']:.4f} | Precision: {test_metrics[1]['Precision']:.4f} | Specificity: {test_metrics[1]['Specificity']:.4f}")
        print(
            f"  AR -> IoU: {test_metrics[2]['IoU']:.4f} | Recall: {test_metrics[2]['Recall']:.4f} | Precision: {test_metrics[2]['Precision']:.4f} | Specificity: {test_metrics[2]['Specificity']:.4f}\n")

if __name__ == "__main__":
    train()