"""
MNIST data utilities for imbalanced classification experiments.

This module provides helpers to:
- Download MNIST from a modern mirror
- Create a class-imbalanced subset (digits 0, 5, and 8)
- Perform a deterministic train/test split
- Return PyTorch DataLoaders
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision import transforms
from torchvision.datasets import MNIST

# Constants
SEED: int = 42
CLASS_COUNTS: dict[int, int] = {8: 3_500, 0: 1_200, 5: 300}
TEST_FRAC: float = 0.20
_MIRROR: str = "https://ossci-datasets.s3.amazonaws.com/mnist/"  # Updated mirror

# Transform: normalize to standard MNIST mean/std
TX = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

class _MNIST(MNIST):
    """MNIST dataset loader."""
    mirrors = [_MIRROR]


def get_raw_mnist(root: str | Path) -> MNIST:
    """
    Download (if needed) and return the full MNIST training dataset (60k images).

    Parameters:
        root (str | Path): Base directory for storing data.

    Returns:
        MNIST: The full training set.
    """
    root = Path(root).expanduser()
    return _MNIST(root=str(root / "MNIST"), train=True, transform=TX, download=True)


def make_subset(ds: MNIST) -> Subset:
    """
    Create a deterministic, class-imbalanced subset of MNIST.

    Digits included: 0 (1.2k), 5 (300), 8 (3.5k)

    Parameters:
        ds (MNIST): The full training dataset.

    Returns:
        Subset: A 5,000-image imbalanced subset.
    """
    rng = random.Random(SEED)
    idx: list[int] = []
    for digit, n in CLASS_COUNTS.items():
        pool = [i for i, y in enumerate(ds.targets) if int(y) == digit]
        rng.shuffle(pool)
        idx.extend(pool[:n])
    rng.shuffle(idx)
    return Subset(ds, idx)


def split_train_test(subset: Dataset) -> Tuple[Subset, Subset]:
    """
    Split a subset into 80% training and 20% test sets (deterministic).

    Parameters:
        subset (Dataset): Dataset to split.

    Returns:
        Tuple[Subset, Subset]: (train_set, test_set)
    """
    test_len = int(len(subset) * TEST_FRAC)
    train_len = len(subset) - test_len
    gen = torch.Generator().manual_seed(SEED)
    return random_split(subset, [train_len, test_len], generator=gen)  # type: ignore[return-value]


def build_dataloaders(
    data_dir: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Convenience function to prepare train/test DataLoaders from raw MNIST.

    This runs:
        - download → subset → split → wrap in DataLoaders

    Parameters:
        data_dir (str | Path): Directory to store/download MNIST.
        batch_size (int): Samples per batch (default: 64).
        num_workers (int): Number of DataLoader workers (default: 0).

    Returns:
        Tuple[DataLoader, DataLoader]: (train_loader, test_loader)
    """
    raw = get_raw_mnist(data_dir)
    train_ds, test_ds = split_train_test(make_subset(raw))
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_dl, test_dl
