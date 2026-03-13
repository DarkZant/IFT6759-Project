import torch
from torch.utils.data import Dataset
import glob
import xarray as xr
import numpy as np
import json
import os
from tqdm import tqdm

class ClimateNetDataset(Dataset):
    def __init__(self, data, folder, time_steps=10, selected_channels=None, train_folder=None):
        self.data = data
        stats_folder = train_folder if train_folder is not None else folder
        first_file = sorted(glob.glob(stats_folder + '/*.nc'))[0]
        ds = xr.open_dataset(first_file, engine='h5netcdf')
        all_channels = [var for var in ds.data_vars if var != 'LABELS']
        ds.close()
        if selected_channels is not None:
            unknown = [c for c in selected_channels if c not in all_channels]
            assert not unknown, f"Unknown channels: {unknown}. Available: {all_channels}"
            self.channels = selected_channels
        else:
            self.channels = all_channels
        self.time_steps = time_steps
        self.means, self.stds = self.compute_mean_per_variable(stats_folder, cache_stats='data/stats.json')

    def __len__(self):
        return len(self.data) - self.time_steps + 1

    def __getitem__(self, idx):
        files = self.data[idx:idx+self.time_steps]
        stacked_data = []
        stacked_label = []
        mean_arr = np.array([self.means[ch] for ch in self.channels])
        std_arr  = np.array([self.stds[ch]  for ch in self.channels])
        for file in files:
            ds = xr.open_dataset(file, engine='h5netcdf')
            data = np.stack([ds[channel].values.squeeze()
                             for channel in self.channels], axis=0)
            label = ds['LABELS'].values

            normalized_data = (data - mean_arr[:, None, None]) / (std_arr[:, None, None] + 1e-8)

            stacked_data.append(normalized_data)
            stacked_label.append(label)
            ds.close()
        return torch.from_numpy(np.stack(stacked_data).astype(np.float32)), torch.from_numpy(stacked_label[-1].astype(np.int64))
        

    def compute_mean_per_variable(self, folder, cache_stats='data/stats.json'):
        # Load existing cache if present
        if os.path.exists(cache_stats):
            with open(cache_stats, 'r') as f:
                stats = json.load(f)
        else:
            stats = {'means': {}, 'stds': {}}

        # Only compute channels missing from cache
        missing = [ch for ch in self.channels if ch not in stats['means']]
        if missing:
            print(f"Computing stats for {len(missing)} channel(s) in a single pass: {missing}")
            # Online accumulators: sum and sum-of-squares per channel
            sums    = {ch: 0.0 for ch in missing}
            sums_sq = {ch: 0.0 for ch in missing}
            counts  = {ch: 0   for ch in missing}

            files = sorted(glob.glob(folder + '/*.nc'))
            for file in tqdm(files, desc="stats pass", unit="file"):
                tqdm.write(f"  {os.path.basename(file)}")
                ds = xr.open_dataset(file, engine='h5netcdf')
                for ch in missing:
                    data = ds[ch].values.flatten().astype(np.float64)
                    sums[ch]    += data.sum()
                    sums_sq[ch] += (data ** 2).sum()
                    counts[ch]  += data.size
                ds.close()

            for ch in missing:
                mean = sums[ch] / counts[ch]
                std  = np.sqrt(sums_sq[ch] / counts[ch] - mean ** 2)
                stats['means'][ch] = float(mean)
                stats['stds'][ch]  = float(std)

        if missing:
            with open(cache_stats, 'w') as f:
                json.dump(stats, f, indent=2)

        return stats['means'], stats['stds']

    def compute_classes_weights(self, cache_stats='data/stats.json'):
        if os.path.exists(cache_stats):
            with open(cache_stats, 'r') as f:
                stats = json.load(f)
            if 'class_weights' in stats:
                return {int(k): v for k, v in stats['class_weights'].items()}

        class_counts = np.zeros(3, dtype=np.int64)
        for file in self.data:
            ds = xr.open_dataset(file, engine='h5netcdf')
            labels = ds['LABELS'].values.flatten().astype(int)
            counts = np.bincount(labels, minlength=3)
            class_counts += counts
            ds.close()
        total_pixels = class_counts.sum()
        class_weight = {i: total_pixels / (3 * count)
                        for i, count in enumerate(class_counts) if count > 0}

        if os.path.exists(cache_stats):
            with open(cache_stats, 'r') as f:
                stats = json.load(f)
        else:
            stats = {'means': {}, 'stds': {}}
        stats['class_weights'] = class_weight
        with open(cache_stats, 'w') as f:
            json.dump(stats, f, indent=2)

        return class_weight

if __name__ == "__main__":
    FOLDER = 'data/climatenet_engineered/train'
    files = sorted(glob.glob(FOLDER + '/*.nc'))
    cn = ClimateNetDataset(files, FOLDER, selected_channels=None)
    print("Stats saved to data/stats.json")
