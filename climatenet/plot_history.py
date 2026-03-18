import json
from os import path
import numpy as np
import matplotlib
matplotlib.use("Agg")   # IMPORTANT for HPC
import matplotlib.pyplot as plt

output_dir = '/project/def-sponsor00/remilalonde/outputs/climatenet_run'
history_path = path.join(output_dir, "training_history.json")

# -------------------------
# Load history
# -------------------------
with open(history_path, "r") as f:
    history = json.load(f)

print("History loaded successfully.")

# -------------------------
# Loss Curve
# -------------------------
plt.figure()
plt.plot(history["train_losses"], label="Train Loss")
plt.plot(history["val_losses"], label="Val Loss")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Loss Curve")
plt.tight_layout()
plt.savefig(path.join(output_dir, "loss_curve.png"))
plt.show()

# -------------------------
# Mean IoU Curve
# -------------------------
plt.figure()
plt.plot(history["train_mean_ious"], label="Train mIoU")
plt.plot(history["val_mean_ious"], label="Val mIoU")
plt.legend()
plt.xlabel("Epoch")
plt.ylabel("Mean IoU")
plt.title("Mean IoU Curve")
plt.tight_layout()
plt.savefig(path.join(output_dir, "miou_curve.png"))
plt.show()

# -------------------------
# IoU per Class
# -------------------------
class_names = ["Background", "TC", "AR"]

train_ious = np.array(history["train_ious_per_class"])
val_ious = np.array(history["val_ious_per_class"])

plt.figure()

for c in range(len(train_ious)):
    plt.plot(train_ious[c], label=f"Train {class_names[c]}")
    #plt.plot(val_ious[c], linestyle="--", label=f"Val {class_names[c]}")

plt.legend()
plt.xlabel("Epoch")
plt.ylabel("IoU")
plt.title("IoU per Class")
plt.tight_layout()
plt.savefig(path.join(output_dir, "iou_per_class.png"))
plt.show()

print("Plots saved successfully.")