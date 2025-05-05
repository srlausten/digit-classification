# Digit Classification CLI

A Pakage for classifying digits **0 · 5 · 8** using an imbalanced MNIST subset:
**3,500×8 | 1,200×0 | 300×5** (5,000 images total, CPU-only).

---

### Testing & Coverage Summary

🟢 **Tests Passing**
📈 **Coverage:** [81% – View Report](htmlcov/index.html)

---

## 1. Installation

### Local (Python 3.11+)

Install dependencies with [**pyenv**](https://github.com/pyenv/pyenv) and [**Poetry**](https://python-poetry.org/docs/):

```bash
pyenv install 3.11.12
pipx install poetry==1.8.1

git clone https://github.com/your-org/digit-classification.git
cd digit-classification
poetry install
```

### Docker

Build the container from the included [Dockerfile](https://docs.docker.com/engine/reference/builder/):

```bash
docker build -t digit-classification .
```

---

## 2. CLI Workflow

```bash
# Download the dataset
digit-classification download-data --data-dir ./data

# Train for {n} epochs
digit-classification train \
  --data-dir   /path/to/data/root \
  --output-dir /path/to/your/output \
  --epochs     {n}

# Evaluate on test split
digit-classification evaluate \
  --checkpoint-path ./runs/exp1/last.ckpt \
  --data-dir        ./data
```

### 🐋 Docker (Mounted at Runtime)

Run any command with `./data` mounted as `/cache`:

```bash
docker run --rm -v $(pwd)/data:/cache digit-classification \
  download-data --data-dir /cache
```

Repeat the same `-v $(pwd)/data:/cache` mount for `train`, `evaluate`, etc.

---

## 3. CLI Commands

| Command         | Description                        | Key Flags                                |
| --------------- | ---------------------------------- | ---------------------------------------- |
| `download-data` | Download MNIST subset              | `--data-dir`                             |
| `train`         | Train CNN                   | `--data-dir`, `--output-dir`, `--epochs` |
| `evaluate`      | Evaluate on test set               | `--checkpoint-path`, `--data-dir`        |
| `predict`       | Predict single image probabilities | `--checkpoint-path`, `--input-path`      |

```bash
digit-classification <command> --help
```

---

## 4. Run Tests / Pre-Commit

```bash
poetry run pytest
poetry run pre-commit run --all-files
```

---

## 5. Local Environment Tested On

| Component | Version                                             |
| --------- | --------------------------------------------------- |
| Python    | 3.11.12 ([pyenv](https://github.com/pyenv/pyenv))   |
| Poetry    | 1.8.1 ([docs](https://python-poetry.org/docs/))     |
| Docker    | 24.x+ ([docs](https://docs.docker.com/get-docker/)) |
| OS        | macOS 13 (Intel)                                    |

---

## License

MIT License © 2025 [Sam Lausten](https://github.com/srlausten)
