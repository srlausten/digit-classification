import torch
from digit_classification.model import DigitClassifier


def test_forward_output_shape():
    model = DigitClassifier()
    x = torch.randn(8, 1, 28, 28)  # batch of 8 grayscale MNIST images
    logits = model(x)
    assert logits.shape == (8, 3)  # 3 classes: 0, 5, 8


def test_predict_step_output_format():
    model = DigitClassifier()
    model.eval()

    x = torch.randn(4, 1, 28, 28)
    dummy_labels = torch.tensor([0, 5, 8, 0])
    preds = model.predict_step((x, dummy_labels), 0)

    assert isinstance(preds, list)
    assert len(preds) == 4
    for pred in preds:
        assert isinstance(pred, dict)
        assert set(pred.keys()) == set([0, 5, 8])
        for prob in pred.values():
            assert 0.0 <= prob <= 1.0
