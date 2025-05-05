import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from torchvision.datasets import MNIST

from digit_classification.constants import TX, SEED, CLASS_COUNTS, TEST_FRAC


def get_raw_mnist(root: str | Path) -> MNIST:
    root = Path(root).expanduser()
    return MNIST(root=str(root), train=True, transform=TX, download=True)


def make_subset(ds: MNIST) -> Subset:
    """
    Create a deterministic, class-imbalanced subset of MNIST.

    Digits included: 0 (1.2k), 5 (300), 8 (3.5k)

    Parameters:
        ds (MNIST): The full training dataset.

    Returns:
        Subset: A 5,000-image subset.
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
    return random_split(subset, [train_len, test_len], generator=gen)  # type: ignore


def build_dataloaders(
    data_dir: str | Path,
    *,
    batch_size: int = 64,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Function to prepare train/test DataLoaders from raw MNIST.

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
    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dl = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )
    return train_dl, test_dl
