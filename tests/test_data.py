import tempfile

from torch.utils.data import Subset
from digit_classification.data import (
    get_raw_mnist,
    make_subset,
    split_train_test,
    build_dataloaders,
)


def test_get_raw_mnist_downloads():
    with tempfile.TemporaryDirectory() as tmp:
        ds = get_raw_mnist(tmp)
        assert len(ds) == 60_000
        assert ds.data.shape == (60_000, 28, 28)


def test_make_subset_is_correct_size():
    with tempfile.TemporaryDirectory() as tmp:
        ds = get_raw_mnist(tmp)
        subset = make_subset(ds)
        assert isinstance(subset, Subset)
        assert len(subset) == 5_000


def test_split_train_test_ratio():
    with tempfile.TemporaryDirectory() as tmp:
        ds = get_raw_mnist(tmp)
        subset = make_subset(ds)
        train, test = split_train_test(subset)
        assert len(train) == 4000
        assert len(test) == 1000


def test_build_dataloaders_shapes():
    with tempfile.TemporaryDirectory() as tmp:
        train_dl, test_dl = build_dataloaders(tmp, batch_size=32, num_workers=0)
        x_batch, y_batch = next(iter(train_dl))
        assert x_batch.shape == (32, 1, 28, 28)
        assert y_batch.shape == (32,)
