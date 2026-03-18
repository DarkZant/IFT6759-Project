import xarray as xr
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from os import path

BASE_OUTPUT_DIR        = 'd:/Universite/Montreal/Project/IFT6759-Project/ClimateNet/outputs/cgnet_4ch'
UNCERTAINTY_OUTPUT_DIR = 'd:/Universite/Montreal/Project/IFT6759-Project/hybrid/outputs/hybrid/hybrid_64'
TEST_PATH              = 'd:/Universite/Montreal/Project/IFT6759-Project/data/climatenet/test'

# Timestep to visualize
T = 6

# ── Load data ────────────────────────────────────────────────────────────────
base_preds  = xr.open_dataarray(path.join(BASE_OUTPUT_DIR,        'predictions.nc'))
unc_preds   = xr.open_dataarray(path.join(UNCERTAINTY_OUTPUT_DIR, 'predictions.nc'))
uncertainty = xr.open_dataarray(path.join(UNCERTAINTY_OUTPUT_DIR, 'predictions.nc'))

# Load ground truth from the test .nc file matching timestep T
import glob, os
test_files = sorted(glob.glob(TEST_PATH + '/*.nc'))
gt_ds      = xr.load_dataset(test_files[T])
gt_frame   = gt_ds['LABELS'].values.squeeze()

base_frame = base_preds.isel(time=T).values
unc_frame  = unc_preds.isel(time=T).values
unc_var    = uncertainty.isel(time=T).values

# ── Error maps ───────────────────────────────────────────────────────────────
base_errors = (base_frame != gt_frame).astype(float)
unc_errors  = (unc_frame  != gt_frame).astype(float)

# ── Plot ─────────────────────────────────────────────────────────────────────
cmap_seg = matplotlib.colors.ListedColormap(['white', 'red', 'blue'])

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle(f'Timestep {T} — Base vs Uncertainty model', fontsize=14)

# Row 1: segmentation maps
for ax, data, title in zip(
    axes[0],
    [gt_frame, base_frame, unc_frame],
    ['Ground Truth', 'Base model (4ch)', 'Uncertainty model (mean pred)']
):
    im = ax.imshow(data, cmap=cmap_seg, vmin=0, vmax=2, interpolation='none')
    ax.set_title(title)
    plt.colorbar(im, ax=ax, ticks=[0, 1, 2]).set_ticklabels(['BG', 'TC', 'AR'])

# Row 2: error maps + uncertainty
im = axes[1][0].imshow(base_errors, cmap='Reds', vmin=0, vmax=1, interpolation='none')
axes[1][0].set_title(f'Base errors  ({base_errors.mean()*100:.1f}% wrong)')
plt.colorbar(im, ax=axes[1][0])

im = axes[1][1].imshow(unc_errors, cmap='Reds', vmin=0, vmax=1, interpolation='none')
axes[1][1].set_title(f'Uncertainty model errors  ({unc_errors.mean()*100:.1f}% wrong)')
plt.colorbar(im, ax=axes[1][1])

im = axes[1][2].imshow(unc_var, cmap='hot', interpolation='none')
axes[1][2].set_title('Uncertainty map (variance)')
plt.colorbar(im, ax=axes[1][2], label='Variance')

plt.tight_layout()

out_path = path.join(UNCERTAINTY_OUTPUT_DIR, f'comparison_t{T}.png')
plt.savefig(out_path, dpi=150)
print(f"Saved to {out_path}")
