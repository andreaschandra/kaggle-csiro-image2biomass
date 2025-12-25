"""Unit tests for csiro_biomass.config module."""

from unittest.mock import patch

import pytest
import yaml

from csiro_biomass.config import (
    Config,
    Dataset,
    FeatureExtractor,
    General,
    Loss,
    Optimizer,
    Scheduler,
    Trainer,
)


class TestGeneral:
    """Tests for General dataclass."""

    def test_general_default_values(self):
        """Test General with default values."""
        general = General()

        assert general.seed == 42
        assert general.model_dir == "models/"
        assert general.img_dir is None
        assert general.kfolds == 1
        assert general.device in ["cuda", "cpu"]

    def test_general_custom_values(self):
        """Test General with custom values."""
        general = General(seed=123, model_dir="custom_models/", img_dir="images/", kfolds=5)

        assert general.seed == 123
        assert general.model_dir == "custom_models/"
        assert general.img_dir == "images/"
        assert general.kfolds == 5

    def test_general_post_init_creates_model_dir(self, tmp_path, monkeypatch):
        """Test that __post_init__ creates model directory."""
        monkeypatch.chdir(tmp_path)
        model_dir = tmp_path / "test_models"

        _ = General(model_dir=str(model_dir))

        assert model_dir.exists()

    def test_general_post_init_existing_model_dir(self, tmp_path):
        """Test __post_init__ with existing model directory."""
        model_dir = tmp_path / "existing_models"
        model_dir.mkdir()

        _ = General(model_dir=str(model_dir))

        assert model_dir.exists()

    @patch("torch.cuda.is_available", return_value=True)
    def test_general_device_cuda_available(self, mock_cuda):
        """Test device is set to cuda when available."""
        general = General()

        assert general.device == "cuda"

    @patch("torch.cuda.is_available", return_value=False)
    def test_general_device_cpu_only(self, mock_cuda):
        """Test device is set to cpu when cuda unavailable."""
        general = General()

        assert general.device == "cpu"


class TestFeatureExtractor:
    """Tests for FeatureExtractor dataclass."""

    def test_feature_extractor_default(self):
        """Test FeatureExtractor with default values."""
        fe = FeatureExtractor()

        assert fe.model is None

    def test_feature_extractor_custom_model(self):
        """Test FeatureExtractor with custom model."""
        fe = FeatureExtractor(model="facebook/dinov2-base")

        assert fe.model == "facebook/dinov2-base"


class TestDataset:
    """Tests for Dataset dataclass."""

    def test_dataset_default(self):
        """Test Dataset with default values."""
        dataset = Dataset()

        assert dataset.train is None
        assert dataset.valid is None
        assert dataset.kfolds == 1

    def test_dataset_custom_values(self):
        """Test Dataset with custom values."""
        dataset = Dataset(train="train.csv", valid="valid.csv", kfolds=5)

        assert dataset.train == "train.csv"
        assert dataset.valid == "valid.csv"
        assert dataset.kfolds == 5


class TestTrainer:
    """Tests for Trainer dataclass."""

    def test_trainer_default(self):
        """Test Trainer with default values."""
        trainer = Trainer()

        assert trainer.epoch == 100
        assert trainer.batch_size == 32
        assert trainer.shuffle is True

    def test_trainer_custom_values(self):
        """Test Trainer with custom values."""
        trainer = Trainer(epoch=50, batch_size=16, shuffle=False)

        assert trainer.epoch == 50
        assert trainer.batch_size == 16
        assert trainer.shuffle is False


class TestOptimizer:
    """Tests for Optimizer dataclass."""

    def test_optimizer_default(self):
        """Test Optimizer with default values."""
        optimizer = Optimizer()

        assert optimizer.type is None
        assert optimizer.params is None

    def test_optimizer_custom_values(self):
        """Test Optimizer with custom values."""
        optimizer = Optimizer(type="AdamW", params={"lr": 0.001})

        assert optimizer.type == "AdamW"
        assert optimizer.params == {"lr": 0.001}


class TestScheduler:
    """Tests for Scheduler dataclass."""

    def test_scheduler_default(self):
        """Test Scheduler with default values."""
        scheduler = Scheduler()

        assert scheduler.type is None
        assert scheduler.params is None

    def test_scheduler_custom_values(self):
        """Test Scheduler with custom values."""
        scheduler = Scheduler(type="CosineAnnealingLR", params={"T_max": 10})

        assert scheduler.type == "CosineAnnealingLR"
        assert scheduler.params == {"T_max": 10}


class TestLoss:
    """Tests for Loss dataclass."""

    def test_loss_default(self):
        """Test Loss with default values."""
        loss = Loss()

        assert loss.type is None
        assert loss.params is None

    def test_loss_custom_values(self):
        """Test Loss with custom values."""
        loss = Loss(type="WeightedMSELoss", params={"reduction": "mean"})

        assert loss.type == "WeightedMSELoss"
        assert loss.params == {"reduction": "mean"}


class TestConfig:
    """Tests for Config dataclass."""

    def test_config_load_from_file(self, sample_config_yaml, tmp_path, monkeypatch):
        """Test Config.load_from_file method."""
        monkeypatch.chdir(tmp_path)

        Config.load_from_file(sample_config_yaml)

        assert Config.general.seed == 42
        assert Config.general.model_dir == "models/"
        assert Config.feature_extractor.model == "facebook/dinov2-base"
        assert Config.dataset.train == "data/train.csv"
        assert Config.trainer.epoch == 10
        assert Config.trainer.batch_size == 8
        assert Config.optimizer.type == "AdamW"
        assert Config.scheduler.type == "CosineAnnealingLR"
        assert Config.loss.type == "WeightedHuberLoss"

    def test_config_load_from_file_partial(self, tmp_path, monkeypatch):
        """Test Config.load_from_file with partial configuration."""
        monkeypatch.chdir(tmp_path)
        partial_config = {
            "general": {"seed": 100},
            "trainer": {"epoch": 50},
        }
        config_file = tmp_path / "partial_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(partial_config, f)

        Config.load_from_file(config_file)

        assert Config.general.seed == 100
        assert Config.trainer.epoch == 50

    def test_config_load_from_file_empty_sections(self, tmp_path, monkeypatch):
        """Test Config.load_from_file with empty sections."""
        monkeypatch.chdir(tmp_path)
        config = {"general": {}, "trainer": {}}
        config_file = tmp_path / "empty_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        Config.load_from_file(config_file)

        # Should use default values
        assert Config.general.seed == 42
        assert Config.trainer.epoch == 100

    def test_config_load_from_file_missing_sections(self, tmp_path, monkeypatch):
        """Test Config.load_from_file with missing sections."""
        monkeypatch.chdir(tmp_path)
        config = {"general": {"seed": 200}}
        config_file = tmp_path / "missing_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config, f)

        Config.load_from_file(config_file)

        assert Config.general.seed == 200
        # Other sections should get default values
        assert Config.feature_extractor.model is None

    def test_config_load_from_file_nonexistent(self):
        """Test Config.load_from_file with non-existent file."""
        with pytest.raises(FileNotFoundError):
            Config.load_from_file("nonexistent.yaml")

    def test_config_load_from_file_all_fields(self, tmp_path, monkeypatch):
        """Test Config.load_from_file loads all fields correctly."""
        monkeypatch.chdir(tmp_path)
        full_config = {
            "general": {
                "seed": 999,
                "model_dir": "my_models/",
                "img_dir": "my_images/",
                "kfolds": 10,
            },
            "feature_extractor": {"model": "test-model"},
            "dataset": {"train": "train.csv", "valid": "valid.csv", "kfolds": 10},
            "trainer": {"epoch": 200, "batch_size": 64, "shuffle": False},
            "optimizer": {"type": "SGD", "params": {"lr": 0.01, "momentum": 0.9}},
            "scheduler": {"type": "StepLR", "params": {"step_size": 30}},
            "loss": {"type": "MSELoss", "params": {}},
        }
        config_file = tmp_path / "full_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(full_config, f)

        Config.load_from_file(config_file)

        assert Config.general.seed == 999
        assert Config.general.model_dir == "my_models/"
        assert Config.general.img_dir == "my_images/"
        assert Config.general.kfolds == 10
        assert Config.feature_extractor.model == "test-model"
        assert Config.dataset.train == "train.csv"
        assert Config.dataset.valid == "valid.csv"
        assert Config.dataset.kfolds == 10
        assert Config.trainer.epoch == 200
        assert Config.trainer.batch_size == 64
        assert Config.trainer.shuffle is False
        assert Config.optimizer.type == "SGD"
        assert Config.optimizer.params["lr"] == 0.01
        assert Config.scheduler.type == "StepLR"
        assert Config.loss.type == "MSELoss"


class TestConfigIntegration:
    """Integration tests for Config."""

    def test_config_workflow(self, tmp_path, monkeypatch):
        """Test complete config loading workflow."""
        monkeypatch.chdir(tmp_path)

        # Create config file
        config_dict = {
            "general": {"seed": 42, "model_dir": str(tmp_path / "models")},
            "trainer": {"epoch": 5, "batch_size": 4},
        }
        config_file = tmp_path / "workflow_config.yaml"
        with open(config_file, "w", encoding="utf-8") as f:
            yaml.dump(config_dict, f)

        # Load config
        Config.load_from_file(config_file)

        # Verify all components loaded
        assert Config.general.seed == 42
        assert Config.trainer.epoch == 5
        assert (tmp_path / "models").exists()
