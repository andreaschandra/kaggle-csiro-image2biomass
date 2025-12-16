# CSIRO Image2Biomass Kaggle Competition

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow template
├── configs/                    # Configuration files
├── data/
│   ├── raw/                   # Raw data from Kaggle
│   ├── processed/             # Processed/transformed data
│   └── external/              # External data sources
├── notebooks/                 # Jupyter notebooks for exploration
├── outputs/
│   ├── models/                # Saved model checkpoints
│   ├── submissions/           # Kaggle submission files
│   └── figures/               # Generated plots and visualizations
├── src/
│   └── csiro_biomass/
│       ├── models/            # Model architectures
│       ├── features/          # Feature engineering
│       └── utils/             # Utility functions
├── pyproject.toml             # Project configuration and dependencies
└── README.md

```

## Setup

This project uses [uv](https://github.com/astral-sh/uv) for fast Python package management.

### Install uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Install dependencies

```bash
# Install base dependencies
uv sync

# Install with optional dependencies
uv sync --extra dev      # Development tools
uv sync --extra ml       # Machine learning frameworks (PyTorch, Lightning, etc.)
uv sync --extra cv       # Computer vision libraries
uv sync --all-extras     # Install everything
```

### Run commands

```bash
# Run Python scripts
uv run python src/csiro_biomass/train.py

# Run Jupyter notebooks
uv run jupyter lab

# Run tests
uv run pytest

# Run linting
uv run ruff check src/
```

## Getting Started

1. Download competition data from Kaggle
2. Place raw data in `data/raw/`
3. Start exploring in `notebooks/`
4. Build models in `src/csiro_biomass/models/`
5. Generate submissions in `outputs/submissions/`

## GitHub Actions

The repository includes a blank GitHub Actions workflow template in `.github/workflows/ci.yml`.
Uncomment and customize the sections as needed for:
- Running tests
- Linting code
- Building and deploying models

## Dependencies

Key dependencies (see `pyproject.toml` for complete list):
- **Core**: numpy, pandas, scikit-learn, matplotlib, seaborn
- **ML** (optional): torch, torchvision, lightning, transformers, timm
- **CV** (optional): opencv-python, pillow, albumentations
- **Dev** (optional): pytest, ruff, black, mypy