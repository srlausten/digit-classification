import pytest

from lightning.pytorch import Trainer
from digit_classification.model import DigitClassifier
from digit_classification.data import build_dataloaders
from digit_classification.evaluation import evaluate_checkpoint


@pytest.fixture(scope="module")
def saved_model_checkpoint(tmp_path_factory):
    tmp_path = tmp_path_factory.mktemp("mnist_eval")
    train_dl, val_dl = build_dataloaders(tmp_path, batch_size=32, num_workers=0)

    model = DigitClassifier()

    trainer = Trainer(
        max_epochs=1,
        accelerator="cpu",
        default_root_dir=tmp_path,
        logger=False,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_dataloaders=train_dl, val_dataloaders=val_dl)

    ckpt_path = tmp_path / "test_model.ckpt"
    trainer.save_checkpoint(str(ckpt_path))

    return ckpt_path, tmp_path


def test_evaluate_checkpoint_returns_expected(saved_model_checkpoint):
    ckpt_path, data_dir = saved_model_checkpoint
    report, accuracy = evaluate_checkpoint(ckpt_path, data_dir)

    assert isinstance(report, str)
    assert "precision" in report
    assert "recall" in report
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0
