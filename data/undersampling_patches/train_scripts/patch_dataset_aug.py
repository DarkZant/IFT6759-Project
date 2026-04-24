"""
patch_dataset.py
────────────────
Dataset PyTorch pour charger les patches .npy générés par preprocess_patches.py.
- Normalisation identique à ClimateDataset.normalize()
- Oversampling des patches TC avec data augmentation (rotations + flips)

Usage dans train_patches.py :
    from patch_dataset import PatchDataset
    from climatenet.utils.utils import Config

    config = Config('config.json')
    train_set = PatchDataset('path/to/patches/train', config, tc_oversample=6)
"""

import numpy as np
import torch
import random
from torch.utils.data import Dataset
import glob
import os


class PatchDataset(Dataset):
    """
    Charge les patches .npy générés par preprocess_patches.py.
    Applique la même normalisation que ClimateDataset.normalize().
    Oversample les patches TC avec data augmentation.

    Chaque fichier .npy contient un dict :
        {'X': (4, 128, 128) float32, 'y': (128, 128) int64}

    Parameters
    ----------
    patch_dir : str
        Dossier contenant les fichiers .npy
    config : Config
        Config du modèle CGNet — contient config.fields avec mean/std
    tc_oversample : int
        Facteur de duplication des patches TC (défaut=1, pas d'oversampling)
        Chaque copie reçoit une augmentation différente (rotation/flip)
    """

    VARIABLES = [
        'TMQ', 'U850', 'V850', 'UBOT', 'VBOT',
        'QREFHT', 'PS', 'PSL', 'T200', 'T500',
        'PRECT', 'TS', 'TREFHT', 'Z1000', 'Z200', 'ZBOT'
    ]

    AUGMENTATIONS = [
        (0, False), # original
        (1, False), # rotation 90
        (2, False), # rotation 180
        (3, False), # rotation 270
        (0, True), # flip horizontal
        (1, True), # rotation 90 + flip
        (2, True), # rotation 180 + flip
        (3, True), # rotation 270 + flip
    ]

    def __init__(self, patch_dir: str, config, tc_oversample: int = 1):

        self.patch_dir = patch_dir
        self.tc_oversample = tc_oversample

        all_files = sorted(glob.glob(os.path.join(patch_dir, '*.npy')))

        if len(all_files) == 0:
            raise FileNotFoundError(f'Aucun fichier .npy trouvé dans {patch_dir}')

        tc_files = [f for f in all_files if '_tc.npy' in f]
        ar_files = [f for f in all_files if '_ar.npy' in f]
        bg_files = [f for f in all_files if '_bg.npy' in f]

        tc_oversampled = []
        n_augs = len(self.AUGMENTATIONS)
        
        for copy_idx in range(tc_oversample):
            aug_idx = copy_idx % n_augs  # cycle sur les 8 augmentations
            for f in tc_files:
                tc_oversampled.append((f, aug_idx))

        # AR et BG: augmentation est None
        ar_entries = [(f, None) for f in ar_files]
        bg_entries = [(f, None) for f in bg_files]

        self.entries = tc_oversampled + ar_entries + bg_entries
        random.shuffle(self.entries)

        # Stats de normalisation depuis config.fields
        means = [config.fields[v]['mean'] for v in self.VARIABLES]
        stds = [config.fields[v]['std']  for v in self.VARIABLES]
        self.means = torch.tensor(means, dtype=torch.float32).view(-1, 1, 1)
        self.stds = torch.tensor(stds,  dtype=torch.float32).view(-1, 1, 1)

        # Affichage
        n_tc_eff = len(tc_files) * tc_oversample
        print(f'[PatchDataset] Chargé depuis {patch_dir}')
        print(f'Fichiers originaux : TC={len(tc_files):,}  AR={len(ar_files):,}  BG={len(bg_files):,}')
        print(f'Après oversampling (TCx{tc_oversample}) :')
        print(f'TC={n_tc_eff:,}  AR={len(ar_files):,}  BG={len(bg_files):,}  Total={len(self.entries):,}')

    def __len__(self):
        return len(self.entries)

    def _apply_augmentation(self, X: torch.Tensor, y: torch.Tensor, aug_idx: int):
        k_rot, do_flip = self.AUGMENTATIONS[aug_idx]

        # Rotation 90 multiplié k
        if k_rot > 0:
            X = torch.rot90(X, k=k_rot, dims=[1, 2])
            y = torch.rot90(y.unsqueeze(0), k=k_rot, dims=[1, 2]).squeeze(0)

        # Flip horizontal
        if do_flip:
            X = torch.flip(X, dims=[2])
            y = torch.flip(y, dims=[1])

        return X, y

    def __getitem__(self, idx: int):
        filepath, aug_idx = self.entries[idx]

        patch = np.load(filepath, allow_pickle=True).item()
        X = torch.from_numpy(patch['X']).float() # (variables, H, W)
        y = torch.from_numpy(patch['y']).long() # (H, W)

        # Normalisation
        X = (X - self.means) / self.stds

        # Augmentation
        if aug_idx is not None:
            X, y = self._apply_augmentation(X, y, aug_idx)

        return X, y

    @staticmethod
    def collate(batch):
        X_list, y_list = zip(*batch)
        return torch.stack(X_list), torch.stack(y_list)
