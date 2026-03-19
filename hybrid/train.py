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
    """
    The .nc filenames encode date and sequence index:
        data-YYYY-MM-DD-HH-1_N.nc
    Files sharing the same date key (YYYY-MM-DD-HH) with sequential _N
    suffixes are truly consecutive timesteps.
    Files on different dates can be weeks apart — feeding them as a
    sequence to a temporal model would be physically meaningless.

    Returns a flat list of files and a list of valid sequence start indices,
    guaranteed to never cross a date-group boundary.
    """
    pattern = re.compile(r'data-(\d{4}-\d{2}-\d{2}-\d{2}-1)_(\d+)\.nc$')
    groups = defaultdict(list)
    for f in all_files:
        m = pattern.search(os.path.basename(f))
        if m:
            groups[m.group(1)].append((int(m.group(2)), f))

    all_files_ordered = []
    valid_starts = []
    for date_key in sorted(groups):
        files_in_group = [f for _, f in sorted(groups[date_key])]
        if len(files_in_group) >= time_steps:
            start = len(all_files_ordered)
            all_files_ordered.extend(files_in_group)
            # Only windows fully inside this group are valid
            for i in range(start, start + len(files_in_group) - time_steps + 1):
                valid_starts.append(i)

    skipped = len(all_files) - sum(
        len(g) for g in groups.values() if len(g) >= time_steps
    )
    print(f"Consecutive-sequence filter: {len(all_files_ordered)} usable files "
          f"({skipped} isolated files dropped), {len(valid_starts)} valid sequences")
    return all_files_ordered, valid_starts

# ── Config ────────────────────────────────────────────────────────────────────
TRAIN_FOLDER   = os.environ.get("TRAIN_FOLDER",   "data/climatenet_engineered/train")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints/hybrid")
NPY_FOLDER     = os.environ.get("NPY_FOLDER",     None)

# Path to a pretrained CGNet weights file.
# Set to None to train the encoder from random init (slower convergence).
# To get weights: train the 2020 baseline first and save with CGNet.save_model().
CGNET_WEIGHTS  = os.environ.get("CGNET_WEIGHTS",  None)

# Set to False to fine-tune CGNet encoder end-to-end (needs more GPU memory)
FREEZE_ENCODER = True

# Same 4 channels as the CGNet baseline for a fair comparison
SELECTED_CHANNELS = ['TMQ', 'U850', 'V850', 'PSL']

HIDDEN_DIM  = 64    # larger than plain ConvLSTM since spatial work is done by CGNet
KERNEL_SIZE = 3
NUM_LAYERS  = 1
NUM_CLASSES = 3
TIME_STEPS  = 3

VAL_SPLIT  = 0.2
BATCH_SIZE = 1
NUM_EPOCHS = 15
LR         = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Data ──────────────────────────────────────────────────────────────────────
SKIP_FILES = {
    "data-2000-04-17-01-1_5.nc",
    "data-2008-10-03-01-1_0.nc",
}

print("Globbing files...", flush=True)
raw_files = [f for f in sorted(glob.glob(TRAIN_FOLDER + "/*.nc"))
             if os.path.basename(f) not in SKIP_FILES]
assert len(raw_files) > 0, f"No .nc files found in {TRAIN_FOLDER}"
print(f"Skipping {len(SKIP_FILES)} known-corrupted file(s).")

# Only keep files that belong to a consecutive group of >= TIME_STEPS frames
all_files, valid_starts = build_consecutive_file_list(raw_files, TIME_STEPS)

split_idx    = int(len(all_files) * (1 - VAL_SPLIT))
train_files  = all_files[:split_idx]
val_files    = all_files[split_idx:]
# Remap valid_starts: train keeps starts < split_idx, val shifts starts by -split_idx
train_starts = [i for i in valid_starts if i + TIME_STEPS - 1 < split_idx]
val_starts   = [i - split_idx for i in valid_starts if i >= split_idx]
print(f"Files — train: {len(train_files)}, val: {len(val_files)}")
print(f"Sequences — train: {len(train_starts)}, val: {len(val_starts)}")

print("Building train dataset...", flush=True)
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
print("Val dataset built.", flush=True)

NUM_WORKERS  = 0
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# ── Model ─────────────────────────────────────────────────────────────────────
# Class weights — from ClimateNet dataset (BG 93.865%, TC 0.462%, AR 5.674%)
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

# ── Training loop ─────────────────────────────────────────────────────────────
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

# ── Save checkpoint ───────────────────────────────────────────────────────────
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

# ── Save one val sample for visualization ─────────────────────────────────────
model.eval()
sample_x, sample_y = val_dataset[0]         # (T, C, H, W), (H, W)
x_in = sample_x.unsqueeze(0).to(DEVICE)     # (1, T, C, H, W)

with torch.no_grad():
    pred_mask = model.predict(x_in)[0].cpu().numpy()  # (H, W)

np.save(os.path.join(CHECKPOINT_DIR, "sample_input.npy"), sample_x[0, 0].numpy())
np.save(os.path.join(CHECKPOINT_DIR, "sample_pred.npy"),  pred_mask)
np.save(os.path.join(CHECKPOINT_DIR, "sample_gt.npy"),    sample_y.numpy())
print(f"Sample arrays saved to {CHECKPOINT_DIR}/")

# ── Save predictions.nc over full test set ────────────────────────────────────
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
