"""
Tests for ClimateNetDataset.
Run from the project root: python -m pytest data/test_dataset.py -v
"""
import glob
import numpy as np
import pytest
import torch

TRAIN_FOLDER = "data/climatenet_engineered/train"
TIME_STEPS = 5
NUM_CHANNELS = 20   # variables excluding LABELS
H, W = 768, 1152


@pytest.fixture(scope="module")
def all_files():
    files = sorted(glob.glob(TRAIN_FOLDER + "/*.nc"))
    assert len(files) > 0, "No .nc files found — check TRAIN_FOLDER path"
    return files


@pytest.fixture(scope="module")
def dataset(all_files):
    from data.dataset import ClimateNetDataset
    return ClimateNetDataset(
        data=all_files,
        folder=TRAIN_FOLDER,
        time_steps=TIME_STEPS,
    )


# ── 1. Initialisation ────────────────────────────────────────────────────────

class TestInit:
    def test_channels_excludes_labels(self, dataset):
        assert "LABELS" not in dataset.channels, \
            "LABELS must not appear in self.channels"

    def test_num_channels(self, dataset):
        assert len(dataset.channels) == NUM_CHANNELS, \
            f"Expected {NUM_CHANNELS} channels, got {len(dataset.channels)}"

    def test_means_keys_match_channels(self, dataset):
        assert set(dataset.means.keys()) == set(dataset.channels), \
            "self.means keys do not match self.channels"

    def test_stds_keys_match_channels(self, dataset):
        assert set(dataset.stds.keys()) == set(dataset.channels), \
            "self.stds keys do not match self.channels"

    def test_means_are_finite(self, dataset):
        for ch, val in dataset.means.items():
            assert np.isfinite(val), f"Mean for {ch} is not finite: {val}"

    def test_stds_are_positive(self, dataset):
        for ch, val in dataset.stds.items():
            assert val > 0, f"Std for {ch} is <= 0: {val}"


# ── 2. Length ─────────────────────────────────────────────────────────────────

class TestLen:
    def test_len_correct(self, dataset, all_files):
        expected = len(all_files) - TIME_STEPS + 1
        assert len(dataset) == expected, \
            f"Expected __len__ == {expected}, got {len(dataset)}"

    def test_len_positive(self, dataset):
        assert len(dataset) > 0


# ── 3. __getitem__ output shapes ─────────────────────────────────────────────

class TestGetItem:
    @pytest.fixture(scope="class")
    def sample(self, dataset):
        return dataset[0]

    def test_returns_tuple_of_two(self, sample):
        assert isinstance(sample, tuple) and len(sample) == 2, \
            "__getitem__ must return a (data, label) tuple"

    def test_data_is_tensor(self, sample):
        x, _ = sample
        assert isinstance(x, torch.Tensor), "data must be a torch.Tensor"

    def test_label_is_tensor(self, sample):
        _, y = sample
        assert isinstance(y, torch.Tensor), "label must be a torch.Tensor"

    def test_data_shape(self, sample):
        x, _ = sample
        assert x.shape == (TIME_STEPS, NUM_CHANNELS, H, W), (
            f"Expected data shape ({TIME_STEPS}, {NUM_CHANNELS}, {H}, {W}), "
            f"got {tuple(x.shape)}\n"
            "HINT: each variable has a time dim (1, H, W) — you may need "
            ".squeeze(0) when stacking channels."
        )

    def test_label_shape(self, sample):
        _, y = sample
        assert y.shape == (H, W), \
            f"Expected label shape ({H}, {W}), got {tuple(y.shape)}"

    def test_label_dtype_is_long(self, sample):
        _, y = sample
        assert y.dtype == torch.long, \
            f"Label dtype must be torch.long, got {y.dtype}"

    def test_data_dtype_is_float32(self, sample):
        x, _ = sample
        assert x.dtype == torch.float32, \
            f"Data dtype must be torch.float32, got {x.dtype}"

    def test_label_values_are_valid_classes(self, sample):
        _, y = sample
        unique = y.unique().tolist()
        assert all(v in [0, 1, 2] for v in unique), \
            f"Label contains unexpected values: {unique}"

    def test_data_has_no_nans(self, sample):
        x, _ = sample
        assert not torch.isnan(x).any(), "data contains NaN values"

    def test_data_has_no_infs(self, sample):
        x, _ = sample
        assert not torch.isinf(x).any(), "data contains Inf values"

    def test_normalization_reasonable(self, sample):
        """After z-score normalization most values should be within ±10."""
        x, _ = sample
        assert x.abs().max().item() < 100, \
            "Normalized values seem very large — check normalization"

    def test_last_index(self, dataset):
        """Edge case: last valid index should not raise."""
        last_idx = len(dataset) - 1
        x, y = dataset[last_idx]
        assert x.shape[0] == TIME_STEPS


# ── 4. DataLoader compatibility ───────────────────────────────────────────────

class TestDataLoader:
    def test_dataloader_batch_shape(self, dataset):
        from torch.utils.data import DataLoader
        loader = DataLoader(dataset, batch_size=2, shuffle=False)
        x_batch, y_batch = next(iter(loader))
        assert x_batch.shape == (2, TIME_STEPS, NUM_CHANNELS, H, W), \
            f"Unexpected batch shape: {tuple(x_batch.shape)}"
        assert y_batch.shape == (2, H, W), \
            f"Unexpected label batch shape: {tuple(y_batch.shape)}"
