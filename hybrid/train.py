import glob
import json
import os
import re
import torch
import numpy as np
import xarray as xr
from torch.utils.data import DataLoader
from collections import defaultdict

from data.dataset import ClimateNetDataset
from hybrid.model import CGNetConvLSTM

print("Script started", flush=True)

def build_consecutive_file_list(all_files, time_steps):
    # extracts date key + N
    pattern = re.compile(r'data-(\d{4}-\d{2}-\d{2}-\d{2}-1)_(\d+)\.nc$')
    # group files by date
    groups = defaultdict(list)
    for f in all_files:
        m = pattern.search(os.path.basename(f))
        if m:
            groups[m.group(1)].append((int(m.group(2)), f))  # (N, path)

    all_files_ordered = []
    valid_starts = []
    for date_key in sorted(groups):  # chronological order
        files_in_group = [f for _, f in sorted(groups[date_key])]  # sort by N
        if len(files_in_group) >= time_steps:  # skip short groups
            start = len(all_files_ordered)  # group start index
            all_files_ordered.extend(files_in_group)
            for i in range(start, start + len(files_in_group) - time_steps + 1):
                valid_starts.append(i)  # windows within group only

    skipped = len(all_files) - sum(
        len(g) for g in groups.values() if len(g) >= time_steps
    )
    print(f"Consecutive-sequence filter: {len(all_files_ordered)} usable files "
          f"({skipped} isolated files dropped), {len(valid_starts)} valid sequences")
    return all_files_ordered, valid_starts

# Config
TRAIN_FOLDER   = os.environ.get("TRAIN_FOLDER",   "data/climatenet_engineered/train")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints/hybrid")
NPY_FOLDER     = os.environ.get("NPY_FOLDER",     None)

CGNET_WEIGHTS  = os.environ.get("CGNET_WEIGHTS",  None)

FREEZE_ENCODER = True

SELECTED_CHANNELS = ['TMQ', 'U850', 'V850', 'PSL']

HIDDEN_DIM  = 64
KERNEL_SIZE = 3
NUM_LAYERS  = 1
NUM_CLASSES = 3
TIME_STEPS  = 3

VAL_SPLIT  = 0.2
BATCH_SIZE = 1
NUM_EPOCHS = 15
LR         = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Data 
SKIP_FILES = {
    "data-2000-04-17-01-1_5.nc",
    "data-2008-10-03-01-1_0.nc",
}

raw_files = [f for f in sorted(glob.glob(TRAIN_FOLDER + "/*.nc"))
             if os.path.basename(f) not in SKIP_FILES] # List of all files


# Only keep files that belong to a consecutive group of >= TIME_STEPS frames
all_files, valid_starts = build_consecutive_file_list(raw_files, TIME_STEPS)

split_idx    = int(len(all_files) * (1 - VAL_SPLIT))
train_files  = all_files[:split_idx]
val_files    = all_files[split_idx:]

train_starts = [i for i in valid_starts if i + TIME_STEPS - 1 < split_idx]
val_starts   = [i - split_idx for i in valid_starts if i >= split_idx]
print(f"Files — train: {len(train_files)}, val: {len(val_files)}")
print(f"Sequences — train: {len(train_starts)}, val: {len(val_starts)}")


train_dataset = ClimateNetDataset(
    data=train_files, folder=TRAIN_FOLDER,
    time_steps=TIME_STEPS, selected_channels=SELECTED_CHANNELS,
    valid_starts=train_starts,
)
print("Train dataset built.", flush=True)
val_dataset = ClimateNetDataset(
    data=val_files, folder=TRAIN_FOLDER,
    time_steps=TIME_STEPS, selected_channels=SELECTED_CHANNELS,
    train_folder=TRAIN_FOLDER,
    valid_starts=val_starts,
)

NUM_WORKERS  = 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Model 
class_weights = torch.tensor([0.355, 72.171, 5.875], dtype=torch.float32)
print(f"Class weights — BG: {class_weights[0]:.3f}, TC: {class_weights[1]:.3f}, AR: {class_weights[2]:.3f}")

print(f"Building model (CGNET_WEIGHTS={CGNET_WEIGHTS})...", flush=True)
model = CGNetConvLSTM(
    hidden_dim=HIDDEN_DIM,
    kernel_size=KERNEL_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    class_weights=class_weights,
    cgnet_weights_path=CGNET_WEIGHTS,
    freeze_encoder=FREEZE_ENCODER,
    channels=len(SELECTED_CHANNELS),
)

print(f"Model Build", flush=True)
model.to(DEVICE)
print(f"Model on {DEVICE}")
print(f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
print(f"Total params:     {sum(p.numel() for p in model.parameters()):,}")

optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=LR
)

# Training loop 
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

history = {
    "train_loss":       [],
    "val_mean_iou":     [],
    "val_iou_bg":       [],
    "val_iou_tc":       [],
    "val_iou_ar":       [],
    "val_mean_recall":  [],
    "val_recall_bg":    [],
    "val_recall_tc":    [],
    "val_recall_ar":    [],
}

print(f"\nStarting training for {NUM_EPOCHS} epochs...")
for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = model.fit(train_loader, optimizer, num_epoch=1, device=DEVICE)
    mean_iou, per_class_iou, mean_recall, per_class_recall = model.evaluate(val_loader, device=DEVICE)

    history["train_loss"].append(train_loss)
    history["val_mean_iou"].append(mean_iou)
    history["val_iou_bg"].append(per_class_iou[0])
    history["val_iou_tc"].append(per_class_iou[1])
    history["val_iou_ar"].append(per_class_iou[2])
    history["val_mean_recall"].append(mean_recall)
    history["val_recall_bg"].append(per_class_recall[0])
    history["val_recall_tc"].append(per_class_recall[1])
    history["val_recall_ar"].append(per_class_recall[2])

    print(f"Epoch {epoch}/{NUM_EPOCHS} — loss: {train_loss:.4f} | "
          f"val mIoU: {mean_iou:.4f} "
          f"(BG {per_class_iou[0]:.4f}, TC {per_class_iou[1]:.4f}, AR {per_class_iou[2]:.4f}) | "
          f"val recall: {mean_recall:.4f} "
          f"(BG {per_class_recall[0]:.4f}, TC {per_class_recall[1]:.4f}, AR {per_class_recall[2]:.4f})")

history_path = os.path.join(CHECKPOINT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"\nHistory saved to {history_path}")

# Save checkpoint 
checkpoint_path = os.path.join(CHECKPOINT_DIR, "cgnet_convlstm_final.pt")
torch.save({
    "model_state_dict":     model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "config": {
        "selected_channels": train_dataset.channels,
        "hidden_dim":        HIDDEN_DIM,
        "kernel_size":       KERNEL_SIZE,
        "num_layers":        NUM_LAYERS,
        "num_classes":       NUM_CLASSES,
        "time_steps":        TIME_STEPS,
        "freeze_encoder":    FREEZE_ENCODER,
        "cgnet_weights":     CGNET_WEIGHTS,
    },
    "val_mean_iou":         mean_iou,
    "val_per_class_iou":    per_class_iou,
    "val_mean_recall":      mean_recall,
    "val_per_class_recall": per_class_recall,
}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")


# Save predictions.nc over full test set 
TEST_FOLDER = os.environ.get("TEST_FOLDER", "/project/def-sponsor00/shared_CN_B/climatenet_engineered/test")
test_files_raw = sorted(glob.glob(TEST_FOLDER + "/*.nc"))
test_files, test_starts = build_consecutive_file_list(test_files_raw, TIME_STEPS)

if len(test_starts) > 0:
    test_dataset = ClimateNetDataset(
        data=test_files, folder=TEST_FOLDER,
        time_steps=TIME_STEPS, selected_channels=SELECTED_CHANNELS,
        train_folder=TRAIN_FOLDER,
        valid_starts=test_starts,
    )
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model.eval()
    all_preds = []
    with torch.no_grad():
        for x, _ in test_loader:
            x = x.to(DEVICE)
            preds = model.predict(x).cpu().numpy()  # (B, H, W)
            all_preds.append(preds)

    all_preds = np.concatenate(all_preds, axis=0)  # (N, H, W)
    da = xr.DataArray(all_preds, dims=['time', 'lat', 'lon'])
    preds_path = os.path.join(CHECKPOINT_DIR, "predictions.nc")
    da.to_netcdf(preds_path)
    print(f"predictions.nc saved to {preds_path}")
else:
    print("No test sequences found — skipping predictions.nc")
