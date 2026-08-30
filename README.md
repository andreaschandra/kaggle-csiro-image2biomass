# 🌾 CSIRO Image2Biomass

> Predicting pasture biomass (clover, dead matter, green matter) from plot images for the
> [CSIRO Image2Biomass](https://www.kaggle.com/competitions/csiro-biomass) Kaggle competition.

![Python](https://img.shields.io/badge/python-3.12-blue)
![uv](https://img.shields.io/badge/package%20manager-uv-de5fe9)
![Ruff](https://img.shields.io/badge/lint-ruff-46a3ff)
![PyTorch](https://img.shields.io/badge/framework-PyTorch-ee4c2c)

## Overview

Each sample is a farm-plot photo; the goal is to regress three dry-weight targets in
grams: **`Dry_Clover_g`**, **`Dry_Dead_g`**, and **`Dry_Green_g`**.

The pipeline is a two-stage transfer-learning approach:

1. **Feature extraction** — each image is split into 9 tiles (8 crops + 1 full resize) and
   passed through a frozen (or partially fine-tuned) vision backbone — either a
   [DINOv2](https://huggingface.co/docs/transformers/model_doc/dinov2) model (via
   `transformers`) or a **ConvNeXtV2** model (via `timm`) — producing a pooled embedding.
2. **Regression head** — a small MLP (or ConvNeXtV2 head) is trained on the embeddings,
   with k-fold cross-validation, early stopping, and a weighted-R² validation metric.

Trained folds are uploaded to both **Hugging Face Hub** and **Kaggle Models**, with runs
tracked in **Weights & Biases**.

## Table of Contents

- [Project Structure](#project-structure)
- [Setup](#setup)
- [Configuration](#configuration)
- [Usage](#usage)
- [Testing & Linting](#testing--linting)
- [Dependencies](#dependencies)

## Project Structure

```
.
├── .github/workflows/ci.yml    # Lint check on push/PR to main
├── configs/                    # Training config YAMLs (git-ignored, empty by default)
├── data/
│   ├── raw/                    # Raw data (auto-downloaded via kagglehub if missing)
│   ├── processed/              # Processed/transformed data
│   └── external/               # External data sources
├── models/                     # Saved model checkpoints, one folder per run
├── notebooks/                  # Exploratory notebooks (DINOv2 + MLP, tiling, augmentation)
├── src/csiro_biomass/
│   ├── config.py                # Dataclass config schema, loaded from YAML
│   ├── dataset.py                # CSIRO dataset: tiling, folds, augmentation, outliers
│   ├── pipeline.py               # Training/validation loop, cross-validation
│   ├── loss.py                   # Weighted MSE / Huber / Tweedie losses
│   ├── metrics.py                # Weighted R² metric
│   ├── optimizer.py              # AdamW wrapper
│   ├── main.py                   # Entry point (train → upload to HF Hub + Kaggle)
│   ├── features/                 # Feature extractors (DINOv2, ConvNeXtV2)
│   ├── models/                   # Regressor heads (MLP, ConvNeXtV2Regressor)
│   └── utils/                    # IO, image processing, logging, HF/Kaggle/W&B helpers
├── tests/                      # Unit and integration tests
├── .env.example                 # Template for KAGGLE_API_TOKEN, HF_TOKEN, WANDB_API_KEY
├── Makefile                     # install / test / run / format / check / clean
└── pyproject.toml               # Project metadata and dependencies
```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### 1. Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Install dependencies

```bash
uv sync                  # base dependencies
uv sync --extra dev      # + development tools (pytest, ruff, black, isort)
uv sync --extra ml       # + PyTorch, torchvision, transformers
uv sync --extra cv       # + opencv-python, pillow, albumentations
uv sync --all-extras     # everything
```

Or simply:

```bash
make install
```

### 3. Configure secrets

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable          | Used for                                   |
|-------------------|---------------------------------------------|
| `KAGGLE_API_TOKEN` | Downloading competition data / uploading models via `kagglehub` |
| `HF_TOKEN`         | Uploading trained model checkpoints to Hugging Face Hub |
| `WANDB_API_KEY`    | Experiment tracking with Weights & Biases   |

## Configuration

Training runs are driven by a YAML config passed to `main.py` (default:
`configs/train.yaml`, which is git-ignored — create your own before running). It maps to
the dataclasses in `src/csiro_biomass/config.py`:

| Section            | Purpose                                                  |
|---------------------|-----------------------------------------------------------|
| `general`           | seed, device, model output dir, image dir, k-folds       |
| `feature_extractor` | which backbone to use (`DinoFeatureExtractor` / `ConvNeXtV2FeatureExtractor`) and pretrained weights |
| `dataset`           | train/test paths, augmentation dir, target transform      |
| `regressor`         | which head to use (`MLP` / `ConvNeXtV2Regressor`)         |
| `trainer`           | epochs, batch size, dataloader workers                    |
| `optimizer`         | optimizer type and params (AdamW)                         |
| `scheduler`         | scheduler params (`ReduceLROnPlateau` on valid R²)         |
| `loss`              | loss type (`WeightedMSELoss` / `WeightedHuberLoss`) and params |
| `huggingface`       | target repo for checkpoint uploads                         |
| `wandb`             | entity / project for run tracking                          |

## Usage

```bash
# Train with the default config (configs/train.yaml)
make run

# Or point at a specific config
uv run python src/csiro_biomass/main.py --config configs/my_run.yaml
```

1. On first run, competition data is auto-downloaded via `kagglehub` if not already present.
2. Explore data and prototype in `notebooks/`.
3. Add new feature extractors under `src/csiro_biomass/features/` or regressor heads under
   `src/csiro_biomass/models/` (see the base classes for the plugin pattern).
4. Trained fold checkpoints are saved to `models/<run_timestamp>/` and pushed to Hugging
   Face Hub and Kaggle Models at the end of a run.

## Testing & Linting

```bash
make test      # pytest --tb=no --disable-warnings
make check      # uv run ruff check src/
make format      # uv run ruff check --fix src/
make clean       # remove caches, models/*, wandb/*
```

Run a single test file or test case with:

```bash
uv run pytest tests/test_dataset.py
uv run pytest tests/test_dataset.py::TestClass::test_case
```

CI (`.github/workflows/ci.yml`) runs `ruff check` on every push/PR to `main`.

## Dependencies

See `pyproject.toml` for the full, versioned list.

- **Core**: numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm, PyYAML
- **ML** *(optional, `--extra ml`)*: torch, torchvision, transformers
- **CV** *(optional, `--extra cv`)*: opencv-python, pillow, albumentations
- **Dev** *(optional, `--extra dev`)*: pytest, ruff, black, isort
- **Integrations**: `kagglehub` (data + model upload), `huggingface-hub` (checkpoint upload), `wandb` (experiment tracking)
