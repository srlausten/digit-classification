from torchvision import transforms

# Random seed for reproducibility
SEED: int = 42

# Per-class sample counts (e.g., for subset selection)
CLASS_COUNTS: dict[int, int] = {
    8: 3_500,
    0: 1_200,
    5: 300,
}

# Test split fraction
TEST_FRAC: float = 0.20

# Raw MNIST download s3 link
_MIRROR: str = "https://ossci-datasets.s3.amazonaws.com/mnist/"

# Default preprocessing transform (standard MNIST normalization)
TX = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
)
