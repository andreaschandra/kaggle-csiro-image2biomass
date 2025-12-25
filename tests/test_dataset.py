"""Unit tests for csiro_biomass.dataset module."""

from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from PIL import Image

from csiro_biomass.dataset import CSIRO


class TestCSIRODataset:
    """Tests for CSIRO dataset."""

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_initialization(
        self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path
    ):
        """Test CSIRO dataset initialization."""
        # Create sample data
        data = {
            "image_path": [f"img_{i}.jpg" for i in range(20)],
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 7)[:20],
            "target": np.random.uniform(10, 100, 20).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(tmp_path)

        dataset = CSIRO(mock_config, mock_feature_extractor)

        assert dataset is not None
        assert dataset.feature_extractor == mock_feature_extractor

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_data_cleanup(self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path):
        """Test CSIRO data cleanup method."""
        data = {
            "image_path": [f"img_{i}.jpg" for i in range(9)],
            "target_name": ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 3,
            "target": [10.0, 20.0, 15.0, 12.0, 22.0, 16.0, 11.0, 21.0, 14.0],
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 2
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(tmp_path)

        dataset = CSIRO(mock_config, mock_feature_extractor)

        # Check that data was cleaned up
        assert dataset is not None

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_target_transform_log1p(
        self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path
    ):
        """Test CSIRO target transformation with log1p."""
        data = {
            "image_path": ["img_0.jpg", "img_1.jpg", "img_2.jpg"],
            "target_name": ["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"],
            "target": [10.0, 20.0, 15.0],
        }
        df = pd.DataFrame(data)
        df = df.pivot(index="image_path", columns="target_name", values="target").reset_index()
        df.columns.name = None
        df = df[["image_path", "Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"]].copy()
        df.iloc[:, 1:4] = df.iloc[:, 1:4].apply(np.log1p)

        dataset_obj = CSIRO.__new__(CSIRO)
        result = dataset_obj.target_transform(df.copy(), method="log1p")

        # Check that log1p was applied
        assert not np.array_equal(result.iloc[:, 1:4].values, df.iloc[:, 1:4].values)

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_set_fold(self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path):
        """Test CSIRO set_fold method."""
        data = {
            "image_path": [f"img_{i}.jpg" for i in range(30)],
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 100, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(tmp_path)

        dataset = CSIRO(mock_config, mock_feature_extractor)
        dataset.set_fold(0)

        assert dataset.dataset is not None

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_set_split_train(
        self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path
    ):
        """Test CSIRO set_split method with train split."""
        data = {
            "image_path": [f"img_{i}.jpg" for i in range(30)],
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 100, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(tmp_path)

        dataset = CSIRO(mock_config, mock_feature_extractor)
        dataset.set_fold(0)
        dataset.set_split("train")

        assert dataset.split_name == "train"
        assert dataset.length > 0

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_set_split_valid(
        self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path
    ):
        """Test CSIRO set_split method with valid split."""
        data = {
            "image_path": [f"img_{i}.jpg" for i in range(30)],
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 100, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(tmp_path)

        dataset = CSIRO(mock_config, mock_feature_extractor)
        dataset.set_fold(0)
        dataset.set_split("valid")

        assert dataset.split_name == "valid"
        assert dataset.length > 0

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_len(self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path):
        """Test CSIRO __len__ method."""
        data = {
            "image_path": [f"img_{i}.jpg" for i in range(30)],
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 100, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(tmp_path)

        dataset = CSIRO(mock_config, mock_feature_extractor)
        dataset.set_fold(0)
        dataset.set_split("train")

        assert len(dataset) > 0
        assert len(dataset) == dataset.length

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_getitem_train(
        self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path
    ):
        """Test CSIRO __getitem__ for training data."""
        # Create test images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (2000, 1000, 3), dtype=np.uint8))
            img.save(img_dir / f"img_{i}.jpg")

        data = {
            "image_path": [f"img_{i}.jpg" for i in range(30)],
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 100, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(img_dir)

        # Setup feature extractor to return tensor
        mock_feature_extractor.return_value = torch.randn(768)

        dataset = CSIRO(mock_config, mock_feature_extractor)
        dataset.set_fold(0)
        dataset.set_split("train")

        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, np.ndarray)
        assert y.shape == (3,)

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_getitem_valid_with_tta(
        self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path
    ):
        """Test CSIRO __getitem__ for validation data with TTA."""
        # Create test images
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        image_list = []
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (2000, 1000, 3), dtype=np.uint8))
            img.save(img_dir / f"img_{i}.jpg")

            image_list.append(f"img_{i}.jpg")
            image_list.append(f"img_{i}.jpg")
            image_list.append(f"img_{i}.jpg")

        data = {
            "image_path": image_list,
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 100, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(img_dir)

        # Setup feature extractor to return tensor
        mock_feature_extractor.return_value = torch.randn(768)

        dataset = CSIRO(mock_config, mock_feature_extractor)
        dataset.set_fold(0)
        dataset.set_split("valid")

        x, y = dataset[0]

        assert isinstance(x, torch.Tensor)
        assert isinstance(y, np.ndarray)
        # With TTA, x should have multiple augmentations
        assert x.shape[0] == 4  # Number of augmentations

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_get_tta(self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path):
        """Test CSIRO get_tta method."""
        data = {
            "image_path": [f"img_{i}.jpg" for i in range(30)],
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 100, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(tmp_path)

        dataset = CSIRO(mock_config, mock_feature_extractor)

        transforms = dataset.get_tta()

        assert len(transforms) == 4  # HFlip, VFlip, Brightness, Hue


class TestCSIRODatasetIntegration:
    """Integration tests for CSIRO dataset."""

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_full_workflow(
        self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path
    ):
        """Test complete CSIRO dataset workflow."""
        # Create test images
        img_dir = tmp_path / "images"
        img_dir.mkdir()

        image_paths = []
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (2000, 1000, 3), dtype=np.uint8))
            img.save(img_dir / f"img_{i}.jpg")

            image_paths.append(f"img_{i}.jpg")
            image_paths.append(f"img_{i}.jpg")
            image_paths.append(f"img_{i}.jpg")

        data = {
            "image_path": image_paths,
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 100, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 3
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(img_dir)

        mock_feature_extractor.return_value = torch.randn(768)

        # Initialize dataset
        dataset = CSIRO(mock_config, mock_feature_extractor)

        # Test all folds
        for fold in range(3):
            dataset.set_fold(fold)

            # Test train split
            dataset.set_split("train")
            assert len(dataset) > 0

            # Test valid split
            dataset.set_split("valid")
            assert len(dataset) > 0

    @patch("csiro_biomass.dataset.read_csv")
    def test_csiro_with_dataloader(
        self, mock_read_csv, mock_config, mock_feature_extractor, tmp_path
    ):
        """Test CSIRO dataset with PyTorch DataLoader."""
        from torch.utils.data import DataLoader

        # Create test images
        img_dir = tmp_path / "images"
        img_dir.mkdir()
        image_paths = []
        for i in range(10):
            img = Image.fromarray(np.random.randint(0, 255, (2000, 1000, 3), dtype=np.uint8))
            img.save(img_dir / f"img_{i}.jpg")

            image_paths.append(f"img_{i}.jpg")
            image_paths.append(f"img_{i}.jpg")
            image_paths.append(f"img_{i}.jpg")

        data = {
            "image_path": image_paths,
            "target_name": (["Dry_Clover_g", "Dry_Dead_g", "Dry_Green_g"] * 10),
            "target": np.random.uniform(10, 3, 30).tolist(),
        }
        df = pd.DataFrame(data)
        mock_read_csv.return_value = df

        mock_config.dataset.train = str(tmp_path / "train.csv")
        mock_config.dataset.kfolds = 2
        mock_config.general.seed = 42
        mock_config.general.img_dir = str(img_dir)

        mock_feature_extractor.return_value = torch.randn(768)

        dataset = CSIRO(mock_config, mock_feature_extractor)
        dataset.set_fold(0)
        dataset.set_split("train")

        dataloader = DataLoader(dataset, batch_size=2, shuffle=False)

        batch_count = 0
        for x, y in dataloader:
            assert x.shape[0] <= 2  # Batch size
            assert y.shape == (x.shape[0], 3)
            batch_count += 1
            if batch_count >= 2:  # Test a few batches
                break

        assert batch_count > 0
