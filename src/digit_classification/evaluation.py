from pathlib import Path
from typing import Tuple

import torch
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from digit_classification.data import build_dataloaders
from digit_classification.model import DigitClassifier
from digit_classification.constants import _IDX_TO_DIGIT, _DIGITS


def _load_test_loader(data_dir: str | Path, batch_size: int) -> DataLoader:
    """Return only the *test* loader from `build_dataloaders`."""
    _, test_loader = build_dataloaders(data_dir, batch_size=batch_size, num_workers=0)
    return test_loader


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    data_dir: str | Path,
    *,
    batch_size: int = 512,
) -> Tuple[str, float]:
    """
    Evaluate a saved checkpoint on the held-out test split.

    Parameters
    ----------
    checkpoint_path
        Path to a model ``.ckpt`` file.
    data_dir
        Directory containing the raw MNIST files.
    batch_size
        Batch size for the evaluation loader.

    Returns
    -------
    report : str
        Multi-line text from :func:`sklearn.metrics.classification_report`.
    accuracy : float
        Macro accuracy over the three classes.
    """
    test_loader = _load_test_loader(data_dir, batch_size)
    model = DigitClassifier.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    model.eval()

    y_true, y_pred = [], []

    with torch.no_grad():
        for x, y_raw in test_loader:
            logits = model(x)
            pred_idx = torch.argmax(logits, dim=1)
            y_true.extend(y_raw.tolist())
            y_pred.extend([_IDX_TO_DIGIT[int(i)] for i in pred_idx])

    report = classification_report(
        y_true,
        y_pred,
        labels=_DIGITS,
        target_names=[str(d) for d in _DIGITS],
        digits=4,
    )
    accuracy = (torch.tensor(y_true) == torch.tensor(y_pred)).float().mean().item()
    return report, accuracy
