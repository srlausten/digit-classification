from pathlib import Path
from typing import Optional

import lightning as L
import typer
import torch
from PIL import Image
from rich import print

from .data import build_dataloaders, get_raw_mnist
from .evaluation import evaluate_checkpoint
from .model import DigitClassifier
from .constants import TX

app = typer.Typer(help="Digit-classification CLI")


@app.command("download-data")
def download_data(
    data_dir: Path = typer.Option(..., "--data-dir", help="Target data directory."),
) -> None:
    """Download the full 60k-image MNIST training split."""
    get_raw_mnist(data_dir)
    print(f"[green]MNIST downloaded to {data_dir/'MNIST'}[/]")


@app.command("train")
def train(
    data_dir: Path = typer.Option(
        ..., "--data-dir", help="Directory containing MNIST data."
    ),
    output_dir: Path = typer.Option(
        ..., "--output-dir", help="Directory to save checkpoints."
    ),
    epochs: int = typer.Option(
        20, "--epochs", min=1, help="Number of epochs (max 20)."
    ),
    batch_size: int = typer.Option(
        64, "--batch-size", min=1, help="Training batch size."
    ),
    lr: float = typer.Option(1e-3, "--lr", help="Learning rate."),
    seed: Optional[int] = typer.Option(None, "--seed", help="Override RNG seed."),
) -> None:
    """Train DigitClassifier on the subset and save to last.ckpt."""
    if epochs > 20:
        raise typer.BadParameter("`epochs` must be 20 or fewer.")

    if seed is not None:
        L.seed_everything(seed, workers=True)

    train_dl, val_dl = build_dataloaders(
        data_dir,
        batch_size=batch_size,
        num_workers=0,
    )
    model = DigitClassifier(lr=lr)

    trainer = L.Trainer(
        default_root_dir=str(output_dir),
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
        ..., "--checkpoint-path", help="Checkpoint file path."
    ),
    data_dir: Path = typer.Option(
        ..., "--data-dir", help="Root Directory containing MNIST data."
    ),
    batch_size: int = typer.Option(512, "--batch-size", help="Evaluation batch size."),
) -> None:
    """Logs metrics on the 20% test split."""
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
        ..., "--checkpoint-path", help="Checkpoint file path."
    ),
    input_path: Path = typer.Option(
        ..., "--input-path", help="Path to input image file."
    ),
) -> None:
    """Predict the digit (0, 5, 8) for a single 28×28 image file."""
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
