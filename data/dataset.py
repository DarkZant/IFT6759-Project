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
    '''
    Dataset for ClimateNet segmentation with temporal sequences.

    Parameters
    ----------
    data : list[str]
        Sorted list of .nc file paths to use.
    folder : str
        Directory containing the .nc files (used to resolve stats).
    time_steps : int
        Number of consecutive frames per sample.
    selected_channels : list[str] or None
        Subset of ALL_CHANNELS to use. None means all 20.
    train_folder : str or None
        Folder used to compute/load normalization stats.
        Defaults to `folder` when None.
    stats_path : str
        Path to the JSON stats cache.
    preload : bool
        If True, load all frames into RAM at construction time.
    npy_folder : str or None
        Directory with pre-converted .npz files for fast preloading.
    '''

    def __init__(self, data, folder, time_steps=10, selected_channels=None,
                 train_folder=None, stats_path='data/stats.json',
                 preload=False, npy_folder=None, valid_starts=None):
        self.data = data
        self.folder = folder
        self.time_steps = time_steps
        # If provided, only these start indices produce valid (non-boundary-crossing) sequences
        self.valid_starts = valid_starts

        if selected_channels is not None:
            unknown = [c for c in selected_channels if c not in ALL_CHANNELS]
            assert not unknown, f"Unknown channels: {unknown}. Available: {ALL_CHANNELS}"
            self.channels = selected_channels
        else:
            self.channels = ALL_CHANNELS

        stats_folder = train_folder if train_folder is not None else folder
        self.fields = self._load_fields(stats_folder, stats_path)

        self._frames = None
        self._labels = None
        if preload:
            self._preload(npy_folder=npy_folder)

    # ── Normalisation (mirrors 2020 ClimateDataset.normalize) ─────────────────

    def normalize(self, data: np.ndarray) -> np.ndarray:
        '''Normalise (C, H, W) array in-place using per-channel mean/std.'''
        for i, ch in enumerate(self.channels):
            data[i] = (data[i] - self.fields[ch]['mean']) / (self.fields[ch]['std'] + 1e-8)
        return data

    def get_features(self, dataset: xr.Dataset) -> np.ndarray:
        '''Extract selected channels from an xarray Dataset and normalise.
        Returns (C, H, W) float32 array.'''
        data = np.stack([dataset[ch].values.squeeze() for ch in self.channels], axis=0).astype(np.float32)
        return self.normalize(data)

    # ── Dataset interface ──────────────────────────────────────────────────────

    def __len__(self):
        if self.valid_starts is not None:
            return len(self.valid_starts)
        return len(self.data) - self.time_steps + 1

    def __getitem__(self, idx: int):
        start = self.valid_starts[idx] if self.valid_starts is not None else idx
        if self._frames is not None:
            data  = np.stack(self._frames[start:start + self.time_steps])
            label = self._labels[start + self.time_steps - 1]
            return torch.tensor(data.copy()), torch.tensor(label.astype(np.int64).copy())

        frames, labels = [], []
        for file in self.data[start:start + self.time_steps]:
            try:
                ds = xr.load_dataset(file)
            except Exception as e:
                print(f"[WARNING] Corrupted file skipped: {file} ({e})")
                return self.__getitem__((idx + 1) % len(self))
            frames.append(self.get_features(ds))
            labels.append(ds['LABELS'].values.copy())
            ds.close()
            del ds
            gc.collect()

        # (T, C, H, W), label from last frame squeezed to (H, W)
        return (torch.tensor(np.stack(frames).copy()),
                torch.tensor(labels[-1].squeeze().astype(np.int64).copy()))

    # ── Stats / fields ─────────────────────────────────────────────────────────

    def _load_fields(self, folder: str, stats_path: str) -> dict:
        '''Load or compute per-channel {mean, std} stats.
        Returns a dict like Config.fields: {channel: {"mean": ..., "std": ...}}.
        '''
        stats = {'means': {}, 'stds': {}}
        if os.path.exists(stats_path):
            print(f"Loading stats from {stats_path}", flush=True)
            with open(stats_path, 'r') as f:
                stats = json.load(f)
        else:
            print(f"No stats cache found at {stats_path}", flush=True)

        missing = [ch for ch in self.channels if ch not in stats.get('means', {})]
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
                stats.setdefault('means', {})[ch] = float(mean)
                stats.setdefault('stds',  {})[ch] = float(np.sqrt(sums_sq[ch] / counts[ch] - mean ** 2))

            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

        # Convert to Config.fields-style dict: {channel: {"mean": ..., "std": ...}}
        return {ch: {'mean': stats['means'][ch], 'std': stats['stds'][ch]}
                for ch in self.channels}

    def compute_classes_weights(self, stats_path: str = 'data/stats.json') -> dict:
        if os.path.exists(stats_path):
            with open(stats_path, 'r') as f:
                stats = json.load(f)
            if 'class_weights' in stats:
                return {int(k): v for k, v in stats['class_weights'].items()}

        step = max(1, len(self.data) // 30)
        class_counts = np.zeros(3, dtype=np.int64)
        for file in self.data[::step]:
            ds = xr.load_dataset(file)
            class_counts += np.bincount(ds['LABELS'].values.flatten().astype(int), minlength=3)
            ds.close()

        total = class_counts.sum()
        class_weight = {i: total / (3 * c) for i, c in enumerate(class_counts) if c > 0}

        stats = json.load(open(stats_path)) if os.path.exists(stats_path) else {'means': {}, 'stds': {}}
        stats['class_weights'] = class_weight
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        return class_weight

    # ── Preload ────────────────────────────────────────────────────────────────

    def _preload(self, npy_folder=None):
        frames, labels = [], []
        if npy_folder and os.path.isdir(npy_folder):
            print(f"Preloading {len(self.data)} files from .npz cache: {npy_folder}")
            skipped = 0
            for file in tqdm(self.data, desc="preload", unit="file"):
                stem = os.path.splitext(os.path.basename(file))[0]
                npz  = os.path.join(npy_folder, stem + '.npz')
                if not os.path.exists(npz):
                    skipped += 1
                    continue
                arr = np.load(npz)
                frames.append(self.normalize(arr['data'].astype(np.float32)))
                labels.append(arr['label'].astype(np.int32))
            if skipped:
                print(f"Skipped {skipped} files with no .npz.")
        else:
            print(f"Preloading {len(self.data)} files from .nc...")
            for file in tqdm(self.data, desc="preload", unit="file"):
                ds = xr.load_dataset(file)
                frames.append(self.get_features(ds))
                labels.append(ds['LABELS'].values.squeeze().copy().astype(np.int32))
                ds.close()
                del ds
                gc.collect()

        self._frames = frames
        self._labels = labels
        print("Preload complete.")


if __name__ == "__main__":
    FOLDER = 'data/climatenet_engineered/train'
    files = sorted(glob.glob(FOLDER + '/*.nc'))
    cn = ClimateNetDataset(files, FOLDER, selected_channels=None)
    print("Stats saved to data/stats.json")
