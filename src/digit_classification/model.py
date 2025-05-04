import torch
import torch.nn.functional as F
from lightning.pytorch import LightningModule
from torch import nn
from torchmetrics.classification import MulticlassAccuracy


_DIGITS = [0, 5, 8]
_NUM_CLASSES = len(_DIGITS)
_LABEL_TO_IDX = {d: i for i, d in enumerate(_DIGITS)}  # 0->0, 5->1, 8->2
_IDX_TO_LABEL = {v: k for k, v in _LABEL_TO_IDX.items()}


class DigitClassifier(LightningModule):
    """Simple CNN → logits(3) → probabilities(3)."""

    def __init__(self, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()  # logs lr in ckpt

        self.net: nn.Sequential = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),  # 28×28 → 28×28
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 14×14
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),                             # 7×7
            nn.Flatten(),
            nn.Linear(7 * 7 * 64, 128),
            nn.ReLU(),
            nn.Linear(128, _NUM_CLASSES),
        )

        self.acc = MulticlassAccuracy(num_classes=_NUM_CLASSES, average="macro")

    def configure_optimizers(self):  # noqa: D401 – imperative header fine
        """Return the Adam optimizer with the LR from ``self.hparams.lr``."""
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)

    # The forward pass is needed for ``self(x)`` or scripted modules.
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Return *logits* (unnormalized scores), shape (B, 3)."""
        return self.net(x)

    # Shared code for training/validation
    def _step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str):
        x, y_raw = batch
        y = torch.tensor([_LABEL_TO_IDX[int(lbl)] for lbl in y_raw], device=self.device)
        logits = self(x)
        loss = F.cross_entropy(logits, y)
        probs = F.softmax(logits, dim=1)
        acc = self.acc(probs, y)

        self.log(f"{stage}_loss", loss, prog_bar=True)
        self.log(f"{stage}_acc", acc, prog_bar=True)
        return loss

    def training_step(self, batch, _batch_idx):
        """One optimization step on the training set."""
        return self._step(batch, "train")

    def validation_step(self, batch, _batch_idx):
        """Validation metric update (no gradient)."""
        self._step(batch, "val")


    def predict_step(self, batch, _batch_idx, _dataloader_idx: int = 0):
        """
        Given a batch of images, return a list of dicts:

        ``[{0: p0, 5: p5, 8: p8}, …]`` where p? are probabilities ∈ [0, 1].
        """
        images, _ = batch
        probs = F.softmax(self(images), dim=1)  # (B, 3)
        preds = []
        for row in probs:
            preds.append({digit: float(row[idx]) for idx, digit in _IDX_TO_LABEL.items()})
        return preds
