from pathlib import Path
from typing import Optional

import lightning as L
import typer
import torch
from PIL import Image
from rich import print

from lightning.pytorch.loggers import TensorBoardLogger
from digit_classification.data import build_dataloaders, get_raw_mnist
from digit_classification.evaluation import evaluate_checkpoint
from digit_classification.model import DigitClassifier
from digit_classification.constants import TX

app = typer.Typer(help="Digit‑classification CLI")


@app.command("download-data")
def download_data(
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-d", help="Target data directory."
    ),
) -> None:
    """Download the full 60 k‑image MNIST training split."""
    get_raw_mnist(data_dir)
    print(f"[green]MNIST downloaded to {data_dir/'MNIST'}[/]")


@app.command("train")
def train(
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-d", help="Directory containing MNIST data."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", "-o", help="Directory to save checkpoints and logs."
    ),
    epochs: int = typer.Option(
        20, "--epochs", "-e", min=1, help="Number of epochs (max 20)."
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", "-b", min=1, help="Training batch size."
    ),
    lr: float = typer.Option(1e-3, "--lr", "-l", help="Learning rate."),
    num_workers: int = typer.Option(
        0,
        "--num-workers",
        "-w",
        min=0,
        help="Dataloader worker processes.",
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override RNG seed."),
) -> None:
    """Train *DigitClassifier* on the subset."""
    if epochs > 20:
        raise typer.BadParameter("`epochs` must be 20 or fewer.")

    if seed is not None:
        L.seed_everything(seed, workers=True)

    train_dl, val_dl = build_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=num_workers,
    )
    model = DigitClassifier(lr=lr)

    logger = TensorBoardLogger(
        save_dir=str(output_dir),
        name="lightning_logs",
    )

    trainer = L.Trainer(
        default_root_dir=str(output_dir),
        logger=logger,
        accelerator="cpu",
        max_epochs=epochs,
        log_every_n_steps=10,
        enable_model_summary=True,
    )
    trainer.fit(
        model,
        train_dataloaders=train_dl,
        val_dataloaders=val_dl,
    )

    ckpt = output_dir / "last.ckpt"
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(ckpt))
    print(f"[green]Training complete – checkpoint saved to {ckpt}[/]")


@app.command("evaluate")
def evaluate(
    checkpoint_path: Path = typer.Option(
        ..., "--checkpoint-path", "-c", help="Checkpoint file path."
    ),
    data_dir: Path = typer.Option(
        ..., "--data-dir", "-d", help="Root directory containing MNIST data."
    ),
    batch_size: int = typer.Option(
        512, "--batch-size", "-b", help="Evaluation batch size."
    ),
) -> None:
    """Log metrics on the held‑out test split."""
    report, acc = evaluate_checkpoint(
        checkpoint_path,
        data_dir,
        batch_size=batch_size,
    )
    print(report)
    print(f"\nMacro accuracy: {acc:.4f}")


@app.command("predict")
def predict(
    checkpoint_path: Path = typer.Option(
        ..., "--checkpoint-path", "-c", help="Checkpoint file path."
    ),
    input_path: Path = typer.Option(
        ..., "--input-path", "-i", help="Path to input image file."
    ),
) -> None:
    """Predict the digit for a single image."""
    img = Image.open(input_path).convert("L").resize((28, 28))
    tensor = TX(img).unsqueeze(0)

    model = DigitClassifier.load_from_checkpoint(
        checkpoint_path,
        map_location="cpu",
    )
    model.eval()

    with torch.no_grad():
        probs = torch.softmax(model(tensor), dim=1).squeeze().tolist()

    digit_map = {0: 0, 1: 5, 2: 8}
    for idx, p in enumerate(probs):
        print(f"{digit_map[idx]}: {p:.4f}")
    pred = digit_map[max(range(len(probs)), key=probs.__getitem__)]
    print(f"\nPrediction: [bold]{pred}[/]")


if __name__ == "__main__":
    app()
