# CSIRO Image2Biomass Kaggle Competition

## Project Structure

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              # GitHub Actions workflow 
├── configs/                    # Configuration files
├── data/
│   ├── raw/                   # Raw data from Kaggle
│   ├── processed/             # Processed/transformed data
│   └── external/              # External data sources
├── logs/
├── models/
├── notebooks/                 # Jupyter notebooks for exploration
├── src/
│   └── csiro_biomass/
│       ├── models/            # Model architectures
│       ├── features/          # Feature engineering
│       └── utils/             # Utility functions
├── tests/                     # Unit and Integration test
├── .env                       # environment variables
├── .gitignore 
├── Makefile
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
make run

# Format code
make format

# Run linting
uv run ruff check src/
```

## Getting Started

1. Download competition data from Kaggle
2. Place raw data in `data/raw/`
3. Start exploring in `notebooks/`
4. Build models in `src/csiro_biomass/models/`
5. Generate submissions in `outputs/submissions/`

## Dependencies

Key dependencies (see `pyproject.toml` for complete list):
- **Core**: numpy, pandas, scikit-learn, matplotlib, seaborn
- **ML** (optional): torch, torchvision, transformers
- **CV** (optional): opencv-python, pillow, albumentations
- **Dev** (optional): pytest, ruff, black, isort