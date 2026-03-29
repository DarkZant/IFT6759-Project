import glob
import json
import os
import torch
import numpy as np
from torch.utils.data import DataLoader

from data.dataset import ClimateNetDataset
from ConvLSTM.convlstm_cell import ConvLSTM

TRAIN_FOLDER   = os.environ.get("TRAIN_FOLDER", "data/climatenet_engineered/train")
CHECKPOINT_DIR = os.environ.get("CHECKPOINT_DIR", "checkpoints")

#   TMQ, U850, V850, UBOT, VBOT, QREFHT, PS, PSL, T200, T500,
#   PRECT, TS, TREFHT, Z1000, Z200, ZBOT, WS850, WSBOT, VRT850, VRTBOT

SELECTED_CHANNELS = ['TMQ', 'U850', 'V850', 'PSL']

HIDDEN_DIM  = 16
KERNEL_SIZE = 3
NUM_LAYERS  = 1
NUM_CLASSES = 3
TIME_STEPS  = 3

VAL_SPLIT   = 0.2
BATCH_SIZE  = 1
NUM_EPOCHS  = 10
LR          = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SKIP_FILES = {
    "data-2000-04-17-01-1_5.nc",
    "data-2008-10-03-01-1_0.nc",
}

all_files = [f for f in sorted(glob.glob(TRAIN_FOLDER + "/*.nc"))
             if os.path.basename(f) not in SKIP_FILES]

# Not shuffled to keep time coherence
split_idx   = int(len(all_files) * (1 - VAL_SPLIT))
train_files = all_files[:split_idx]
val_files   = all_files[split_idx:]

print(f"Files — train: {len(train_files)}, val: {len(val_files)}")


# Datasets and DataLoaders

train_dataset = ClimateNetDataset(
    data=train_files, folder=TRAIN_FOLDER,
    time_steps=TIME_STEPS, selected_channels=SELECTED_CHANNELS,
)
val_dataset = ClimateNetDataset(
    data=val_files, folder=TRAIN_FOLDER,
    time_steps=TIME_STEPS, selected_channels=SELECTED_CHANNELS,
)


INPUT_DIM = len(train_dataset.channels)
print(f"Using {INPUT_DIM} channels")

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False)
val_loader   = DataLoader(val_dataset,   batch_size=BATCH_SIZE, shuffle=False)

print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}")

# Class weights (BG 93.865%, TC 0.462%, AR 5.674%)
class_weights = torch.tensor([0.355, 72.171, 5.875], dtype=torch.float32)


# Model and optimizer

model = ConvLSTM(
    input_dim=INPUT_DIM,
    hidden_dim=HIDDEN_DIM,
    kernel_size=KERNEL_SIZE,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    class_weights=class_weights,
)
model.to(DEVICE)
print(f"Model on {DEVICE}")

optimizer = torch.optim.Adam(model.parameters(), lr=LR)


# Train

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

print(f"\nStarting training for {NUM_EPOCHS} epochs")
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
          f"val mIoU: {mean_iou:.4f} (BG {per_class_iou[0]:.4f}, TC {per_class_iou[1]:.4f}, AR {per_class_iou[2]:.4f}) | "
          f"val recall: {mean_recall:.4f} (BG {per_class_recall[0]:.4f}, TC {per_class_recall[1]:.4f}, AR {per_class_recall[2]:.4f})")

history_path = os.path.join(CHECKPOINT_DIR, "history.json")
with open(history_path, "w") as f:
    json.dump(history, f, indent=2)
print(f"\nHistory saved to {history_path}")


# Saves

checkpoint_path = os.path.join(CHECKPOINT_DIR, "convlstm_final.pt")
torch.save({
    "model_state_dict":      model.state_dict(),
    "optimizer_state_dict":  optimizer.state_dict(),
    "config": {
        "input_dim":         INPUT_DIM,
        "selected_channels": train_dataset.channels,
        "hidden_dim":        HIDDEN_DIM,
        "kernel_size":       KERNEL_SIZE,
        "num_layers":        NUM_LAYERS,
        "num_classes":       NUM_CLASSES,
        "time_steps":        TIME_STEPS,
    },
    "val_mean_iou":        mean_iou,
    "val_per_class_iou":   per_class_iou,
    "val_mean_recall":     mean_recall,
    "val_per_class_recall": per_class_recall,
}, checkpoint_path)
print(f"Checkpoint saved to {checkpoint_path}")


# # Save one val sample for local visualization

# model.eval()
# sample_x, sample_y = val_dataset[0]          # (T, C, H, W), (H, W)
# x_in = sample_x.unsqueeze(0).to(DEVICE)      # (1, T, C, H, W)

# with torch.no_grad():
#     pred_mask = model.predict(x_in)[0].cpu().numpy()   # (H, W)

# np.save(os.path.join(CHECKPOINT_DIR, "sample_input.npy"),  sample_x[0, 0].numpy())
# np.save(os.path.join(CHECKPOINT_DIR, "sample_pred.npy"),   pred_mask)
# np.save(os.path.join(CHECKPOINT_DIR, "sample_gt.npy"),     sample_y.numpy())
# print(f"Sample arrays saved to {CHECKPOINT_DIR} (input/pred/gt .npy)")
