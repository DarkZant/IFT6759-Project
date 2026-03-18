"""
Plot saved sample arrays from a training run.

Usage:
    python ConvLSTM/plot_sample.py checkpoints/<experiment_name>
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

checkpoint_dir = sys.argv[1] if len(sys.argv) > 1 else "checkpoints/convlstm"

input_img = np.load(os.path.join(checkpoint_dir, "sample_input.npy"))
pred_mask  = np.load(os.path.join(checkpoint_dir, "sample_pred.npy"))
gt_mask    = np.load(os.path.join(checkpoint_dir, "sample_gt.npy"))

cmap = mcolors.ListedColormap(["#1f77b4", "#d62728", "#2ca02c"])  # BG, TC, AR
norm = mcolors.BoundaryNorm([0, 1, 2, 3], cmap.N)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].imshow(input_img, cmap="viridis")
axes[0].set_title("Input: channel 0 (t=0)")
axes[0].axis("off")

im = axes[1].imshow(pred_mask, cmap=cmap, norm=norm)
axes[1].set_title("Predicted mask")
axes[1].axis("off")

axes[2].imshow(gt_mask, cmap=cmap, norm=norm)
axes[2].set_title("Ground truth mask")
axes[2].axis("off")

cbar = fig.colorbar(im, ax=axes, ticks=[0.5, 1.5, 2.5], fraction=0.015, pad=0.02)
cbar.ax.set_yticklabels(["Background", "Tropical Cyclone", "Atmospheric River"])

out_path = os.path.join(checkpoint_dir, "sample_prediction.png")
plt.savefig(out_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"Saved to {out_path}")
