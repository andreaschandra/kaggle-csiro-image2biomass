"""Shared pytest fixtures and configurations for all tests."""

import tempfile
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest
import torch
import yaml


@pytest.fixture
def tmp_dir():
    """Create a temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def sample_config_dict():
    """Sample configuration dictionary."""
    return {
        "general": {
            "seed": 42,
            "model_dir": "models/",
            "img_dir": "data/images/",
            "kfolds": 3,
        },
        "feature_extractor": {"model": "facebook/dinov2-base"},
        "dataset": {"train": "data/train.csv", "valid": "data/valid.csv", "kfolds": 3},
        "trainer": {"epoch": 10, "batch_size": 8, "shuffle": True},
        "optimizer": {"type": "AdamW", "params": {"lr": 0.001}},
        "scheduler": {"type": "CosineAnnealingLR", "params": {"T_max": 10}},
        "loss": {"type": "WeightedHuberLoss", "params": {"delta": 3.0}},
    }


@pytest.fixture
def sample_config_yaml(tmp_path, sample_config_dict):
    """Create a sample YAML config file."""
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w", encoding="utf-8") as f:
        yaml.dump(sample_config_dict, f)
    return config_file


@pytest.fixture
def sample_train_csv(tmp_path):
    """Create a sample training CSV file."""
    csv_file = tmp_path / "train.csv"
    data = {
        "image_path": [f"img_{i}.jpg" for i in range(20)],
        "target_name": ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 6
        + ["Dry_Clover_g", "Dry_Dead_g"],
        "target": np.random.uniform(10, 100, 20).tolist(),
    }
    df = pd.DataFrame(data)
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def sample_images_dir(tmp_path):
    """Create a directory with sample images."""
    img_dir = tmp_path / "images"
    img_dir.mkdir(exist_ok=True)

    # Create dummy image files (just empty files for testing)
    for i in range(10):
        img_file = img_dir / f"img_{i}.jpg"
        img_file.touch()

    return img_dir


@pytest.fixture
def mock_torch_model():
    """Mock PyTorch model."""
    model = MagicMock(spec=torch.nn.Module)
    model.parameters.return_value = [torch.randn(10, 10, requires_grad=True)]
    model.eval.return_value = None
    model.train.return_value = None
    model.to.return_value = model
    model.state_dict.return_value = {"layer1.weight": torch.randn(10, 10)}
    return model


@pytest.fixture
def mock_feature_extractor():
    """Mock feature extractor."""
    extractor = MagicMock()
    extractor.get_embedding_dim.return_value = 768
    extractor.return_value = torch.randn(768)
    return extractor


@pytest.fixture
def mock_dino_processor():
    """Mock DINO image processor."""
    processor = MagicMock()
    processor.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
    processor.to.return_value = processor
    return processor


@pytest.fixture
def mock_dino_model():
    """Mock DINO model."""
    model = MagicMock()
    output = MagicMock()
    output.last_hidden_state = torch.randn(2, 197, 768)  # [batch, patches+cls, dim]
    model.return_value = output
    model.eval.return_value = None
    model.to.return_value = model
    return model


@pytest.fixture
def sample_tensor_batch():
    """Sample batch of tensors."""
    return {
        "inputs": torch.randn(4, 768),
        "targets": torch.randn(4, 3),
    }


@pytest.fixture
def sample_predictions_targets():
    """Sample predictions and targets for metrics testing."""
    pred = torch.tensor([[10.5, 20.3, 15.7], [12.1, 18.5, 14.2], [9.8, 22.1, 16.5]])
    target = torch.tensor([[10.0, 20.0, 15.0], [12.0, 19.0, 14.0], [10.0, 22.0, 16.0]])
    return pred, target


@pytest.fixture
def mock_logger():
    """Mock logger."""
    logger = MagicMock()
    logger.info.return_value = None
    logger.debug.return_value = None
    logger.warning.return_value = None
    logger.error.return_value = None
    return logger


@pytest.fixture
def mock_config():
    """Mock config object."""
    config = MagicMock()

    # General config
    config.general.seed = 42
    config.general.model_dir = "models/"
    config.general.img_dir = "data/images/"
    config.general.device = "cpu"
    config.general.kfolds = 3

    # Feature extractor config
    config.feature_extractor.model = "facebook/dinov2-base"

    # Dataset config
    config.dataset.train = "data/train.csv"
    config.dataset.valid = "data/valid.csv"
    config.dataset.kfolds = 3

    # Trainer config
    config.trainer.epoch = 10
    config.trainer.batch_size = 8
    config.trainer.shuffle = True

    # Optimizer config
    config.optimizer.type = "AdamW"
    config.optimizer.params = {"lr": 0.001}

    # Scheduler config
    config.scheduler.type = "CosineAnnealingLR"
    config.scheduler.params = {"T_max": 10}

    # Loss config
    config.loss.type = "WeightedHuberLoss"
    config.loss.params = {"delta": 3.0}

    return config
