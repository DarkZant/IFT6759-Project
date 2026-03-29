import gc
import torch
from torch.utils.data import Dataset
import glob
import xarray as xr
import numpy as np
import json
import os
from tqdm import tqdm

# All available channels in the engineered dataset
ALL_CHANNELS = [
    'TMQ', 'U850', 'V850', 'UBOT', 'VBOT', 'QREFHT', 'PS', 'PSL',
    'T200', 'T500', 'PRECT', 'TS', 'TREFHT', 'Z1000', 'Z200',
    'ZBOT', 'WS850', 'WSBOT', 'VRT850', 'VRTBOT',
]


class ClimateNetDataset(Dataset):


    def __init__(self, data, folder, time_steps=10, selected_channels=None,
                 train_folder=None, stats_path='data/stats.json', valid_starts=None):
        self.data = data
        self.folder = folder
        self.time_steps = time_steps
        self.valid_starts = valid_starts

        if selected_channels is not None:
            self.channels = selected_channels
        else:
            self.channels = ALL_CHANNELS

        stats_folder = train_folder if train_folder is not None else folder
        self.fields = self._load_fields(stats_folder, stats_path)


    def normalize(self, data: np.ndarray) -> np.ndarray:
        '''Normalise (C, H, W) array in-place using per-channel mean/std.'''
        for i, ch in enumerate(self.channels):
            data[i] = (data[i] - self.fields[ch]['mean']) / (self.fields[ch]['std'] + 1e-8)
        return data

    def get_features(self, dataset: xr.Dataset) -> np.ndarray:
        '''Extract selected channels from an xarray Dataset and normalise.'''
        data = np.stack([dataset[ch].values.squeeze() for ch in self.channels], axis=0).astype(np.float32)
        return self.normalize(data)


    def __len__(self):
        if self.valid_starts is not None:
            return len(self.valid_starts)
        return len(self.data) - self.time_steps + 1

    def __getitem__(self, idx: int):
        start = self.valid_starts[idx] if self.valid_starts is not None else idx

        frames, labels = [], []
        for file in self.data[start:start + self.time_steps]:
            try:
                ds = xr.load_dataset(file) # Open .nc file
            except Exception as e:
                print(f"Corrupted file skipped: {file} ({e})")
                return self.__getitem__((idx + 1) % len(self))
            frames.append(self.get_features(ds))
            labels.append(ds['LABELS'].values.copy())
            ds.close()
            del ds
            gc.collect() # Free memory immediately after loading each file

        return (torch.tensor(np.stack(frames).copy()),
                torch.tensor(labels[-1].squeeze().astype(np.int64).copy()))


    def _load_fields(self, folder: str, stats_path: str) -> dict:
        '''Load or compute per-channel {mean, std} stats '''
        stats = {'means': {}, 'stds': {}}
        if os.path.exists(stats_path): # check if stats cache exists
            print(f"Loading stats from {stats_path}", flush=True)
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        else:
            print(f"No stats cache found at {stats_path}", flush=True)

        missing = [ch for ch in self.channels if ch not in stats.get('means', {})] # array of missing channels for which we need to compute stats
        if missing:
            print(f"Computing stats for {len(missing)} channel(s): {missing}")
            sums, sums_sq, counts = {ch: 0.0 for ch in missing}, {ch: 0.0 for ch in missing}, {ch: 0 for ch in missing}

            for file in tqdm(sorted(glob.glob(folder + '/*.nc')), desc="stats pass", unit="file"):
                ds = xr.load_dataset(file)
                for ch in missing:
                    vals = ds[ch].values.flatten().astype(np.float64)
                    sums[ch]    += vals.sum()
                    sums_sq[ch] += (vals ** 2).sum()
                    counts[ch]  += vals.size
                ds.close()

            for ch in missing:
                mean = sums[ch] / counts[ch]
                stds = np.sqrt(sums_sq[ch] / counts[ch] - mean ** 2)
                if 'means' not in stats:
                    stats['means'] = {}
                if 'stds' not in stats:
                    stats['stds'] = {}
                stats['means'][ch] = float(mean)
                stats['stds'][ch]  = float(stds)

            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

        return {ch: {'mean': stats['means'][ch], 'std': stats['stds'][ch]}
                for ch in self.channels}

