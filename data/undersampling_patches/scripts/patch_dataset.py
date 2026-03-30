"""
patch_dataset.py
────────────────
Dataset PyTorch pour charger les patches .npy générés par preprocess_patches.py.
La normalisation reproduit exactement celle de ClimateDataset.normalize()
en utilisant les stats de config.json.

Usage dans train_patches.py :
    from patch_dataset import PatchDataset
    from climatenet.utils.utils import Config

    config    = Config('config.json')
    train_set = PatchDataset('path/to/patches/train', config)
"""

import numpy as np
import torch
from torch.utils.data import Dataset
import glob
import os


class PatchDataset(Dataset):
    """
    Charge les patches .npy générés par preprocess_patches.py
    et applique la même normalisation que ClimateDataset.normalize().

    Chaque fichier .npy contient un dict :
        {'X': (4, 128, 128) float32, 'y': (128, 128) int64}

    Parameters
    ----------
    patch_dir : str
        Dossier contenant les fichiers .npy
    config : Config
        Config du modèle CGNet — contient config.fields avec mean/std
        par variable, dans le même ordre que preprocess_patches.py
    """

    # Ordre des 4 variables — doit correspondre à preprocess_patches.py
    VARIABLES = ['TMQ', 'U850', 'V850', 'PSL']

    def __init__(self, patch_dir: str, config):
        self.patch_dir = patch_dir

        self.files = sorted(glob.glob(os.path.join(patch_dir, '*.npy')))
        if len(self.files) == 0:
            raise FileNotFoundError(f'Aucun fichier .npy trouvé dans {patch_dir}')

        # Stats de normalisation depuis config.fields
        # Reproduit ClimateDataset.normalize() :
        #   var -= stats['mean']
        #   var /= stats['std']
        means = []
        stds  = []
        for var in self.VARIABLES:
            stats = config.fields[var]
            means.append(stats['mean'])
            stds.append(stats['std'])

        # shape (4, 1, 1) pour broadcaster sur (4, 128, 128)
        self.means = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.stds  = torch.tensor(stds,  dtype=torch.float32).view(-1, 1, 1)

        print(f'[PatchDataset] {len(self.files):,} patches chargés depuis {patch_dir}')
        print(f'  Normalisation : {self.VARIABLES}')
        for var, m, s in zip(self.VARIABLES, means, stds):
            print(f'    {var:<8} mean={m:.4f}  std={s:.4f}')

        # Distribution TC/AR/BG
        tc = sum(1 for f in self.files if '_tc.npy' in f)
        ar = sum(1 for f in self.files if '_ar.npy' in f)
        bg = sum(1 for f in self.files if '_bg.npy' in f)
        print(f'  TC : {tc:>6,}  |  AR : {ar:>6,}  |  BG : {bg:>6,}')

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx: int):
        patch = np.load(self.files[idx], allow_pickle=True).item()

        X = torch.from_numpy(patch['X']).float()   # (4, 128, 128)
        y = torch.from_numpy(patch['y']).long()    # (128, 128)

        # Normalisation identique à ClimateDataset.normalize()
        X = (X - self.means) / self.stds

        return X, y

    @staticmethod
    def collate(batch):
        """Collate function compatible avec DataLoader."""
        X_list, y_list = zip(*batch)
        return torch.stack(X_list), torch.stack(y_list)