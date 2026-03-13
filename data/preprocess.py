"""
Convert .nc files to .npz format (no HDF5 during training).
Run once via ConvLSTM/preprocess.sh before training.
"""
import glob
import os
import numpy as np
import xarray as xr
from tqdm import tqdm

SRC = os.environ.get('TRAIN_FOLDER', 'data/climatenet_engineered/train')
DST = os.environ.get('NPY_FOLDER',   'data/climatenet_npy/train')
CHANNELS = os.environ.get('CHANNELS', 'TMQ,U850,V850,PSL').split(',')

os.makedirs(DST, exist_ok=True)
files = sorted(glob.glob(SRC + '/*.nc'))
print(f"Converting {len(files)} files: {SRC} -> {DST}")
print(f"Channels: {CHANNELS}")

skipped = 0
for f in tqdm(files, desc='converting', unit='file'):
    stem = os.path.splitext(os.path.basename(f))[0]
    out  = os.path.join(DST, stem + '.npz')
    if os.path.exists(out):
        skipped += 1
        continue
    ds    = xr.load_dataset(f)
    data  = np.stack([ds[c].values.squeeze() for c in CHANNELS], axis=0).astype(np.float32)
    label = ds['LABELS'].values.squeeze().astype(np.int8)
    ds.close()
    del ds
    np.savez_compressed(out, data=data, label=label)

print(f"Done. {len(files) - skipped} converted, {skipped} skipped (already exist).")
