"""Configuration classes for CSIRO Biomass project."""

import os
from dataclasses import dataclass

import torch

from csiro_biomass.utils.io import read_yaml


@dataclass
class General:
    """General configuration"""

    seed: int = 42
    model_dir: str = "models/"
    img_dir: str = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    kfolds: int = 1

    def __post_init__(self):
        """Post initialization processing."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        if os.path.exists(self.model_dir) is False:
            os.makedirs(self.model_dir)


@dataclass
class FeatureExtractor:
    """Feature extractor configuration."""

    model: str = None


@dataclass
class Dataset:
    """Dataset configuration."""

    train: str = None
    valid: str = None
    kfolds: int = 1


@dataclass
class Trainer:
    """Trainer configuration."""

    epoch: int = 100
    batch_size: int = 32
    shuffle: bool = True


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

    type: float = None
    params: dict = None


@dataclass
class Config:
    """Main configuration."""

    general: General
    feature_extractor: FeatureExtractor
    dataset: Dataset
    trainer: Trainer
    optimizer: Optimizer
    scheduler: Scheduler
    loss: Loss

    @staticmethod
    def load_from_file(path):
        """Load from a YAML file."""

        config_dict = read_yaml(path)
        Config.general = General(**config_dict.get("general", {}))
        Config.feature_extractor = FeatureExtractor(**config_dict.get("feature_extractor", {}))
        Config.dataset = Dataset(**config_dict.get("dataset", {}))
        Config.trainer = Trainer(**config_dict.get("trainer", {}))
        Config.optimizer = Optimizer(**config_dict.get("optimizer", {}))
        Config.scheduler = Scheduler(**config_dict.get("scheduler", {}))
        Config.loss = Loss(**config_dict.get("loss", {}))

        return Config
