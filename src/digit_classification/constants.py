from torchvision import transforms

# Random seed for reproducibility
SEED: int = 42

# Per-class sample counts (e.g., for subset selection)
CLASS_COUNTS: dict[int, int] = {
    8: 3_500,
    0: 1_200,
    5: 300,
}

_DIGITS = [0, 5, 8]
_IDX_TO_DIGIT = {idx: digit for idx, digit in enumerate(_DIGITS)}
_DIGIT_TO_IDX = {digit: idx for idx, digit in _IDX_TO_DIGIT.items()}

# Test split fraction
TEST_FRAC: float = 0.20

# Raw MNIST download s3 link
_MIRROR: str = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# Default preprocessing transform (standard MNIST normalization)
TX = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
