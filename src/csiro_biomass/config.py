"""Configuration classes for CSIRO Biomass project."""

import os
from dataclasses import asdict, dataclass
from datetime import datetime

import torch

from csiro_biomass.utils.io import read_yaml


@dataclass
class General:
    """General configuration"""

    seed: int = 42
    competition: str = "csiro_biomass"
    model_dir: str = "models/"
    img_dir: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    kfolds: int = 1

    def __post_init__(self):
        """Post initialization processing."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.run_at = datetime.now().strftime("%Y%m%d-%H%M%S")

        if os.path.exists(self.model_dir) is False:
            os.makedirs(self.model_dir)


@dataclass
class FeatureExtractor:
    """Feature extractor configuration."""

    is_enabled: bool = False
    model: str = None
    pretrained_name: str = None


@dataclass
class Dataset:
    """Dataset configuration."""

    train: str = None
    test: str = None
    kfolds: int = 1
    use_train_augmentation: bool = True
    augmented: str = None
    aug_dir: str = None
    target_transform: bool = False


@dataclass
class Regressor:
    model: str


@dataclass
class Trainer:
    """Trainer configuration."""

    epoch: int = 100
    batch_size: int = 32
    shuffle: bool = True
    num_workers: int = 0


@dataclass
class Optimizer:
    """Optimizer configuration."""

    type: float = None
    params: dict = None


@dataclass
class Scheduler:
    """Scheduler configuration."""

    type: float = None
    params: dict = None


@dataclass
class Loss:
    """Loss configuration."""

    type: float
    params: dict


@dataclass
class Hugginface:
    """Huggingface configuration."""

    repo_id: str = None


@dataclass
class Wandb:
    """Weights and Biases configuration."""

    project: str = None
    entity: str = None


@dataclass
class Kaggle:
    """Kaggle configuration."""

    model_name: str = None


@dataclass
class Config:
    """Main configuration."""

    general: General
    feature_extractor: FeatureExtractor
    dataset: Dataset
    regressor: Regressor
    trainer: Trainer
    optimizer: Optimizer
    scheduler: Scheduler
    loss: Loss
    huggingface: Hugginface
    wandb: Wandb

    def to_dict(self) -> dict:
        """Convert config to dictionary recursively."""
        return asdict(self)

    @classmethod
    def load_from_file(cls, path):
        """Load from a YAML file."""

        config_dict = read_yaml(path)
        return cls(
            general=General(**config_dict.get("general", {})),
            feature_extractor=FeatureExtractor(**config_dict.get("feature_extractor", {})),
            dataset=Dataset(**config_dict.get("dataset", {})),
            regressor=Regressor(**config_dict.get("regressor", {})),
            trainer=Trainer(**config_dict.get("trainer", {})),
            optimizer=Optimizer(**config_dict.get("optimizer", {})),
            scheduler=Scheduler(**config_dict.get("scheduler", {})),
            loss=Loss(**config_dict.get("loss", {})),
            huggingface=Hugginface(**config_dict.get("huggingface", {})),
            wandb=Wandb(**config_dict.get("wandb", {})),
        )
