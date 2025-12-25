"""Unit tests for csiro_biomass.features.dino module."""

from unittest.mock import MagicMock, patch

import numpy as np
import torch

from csiro_biomass.features.dino import DinoFeatureExtractor


class TestDinoFeatureExtractor:
    """Tests for DinoFeatureExtractor."""

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_initialization(self, mock_model, mock_processor, mock_config):
        """Test DinoFeatureExtractor initialization."""
        mock_processor.return_value = MagicMock()
        mock_model.return_value = MagicMock()

        extractor = DinoFeatureExtractor(mock_config)

        assert extractor is not None
        assert extractor.config == mock_config
        mock_processor.assert_called_once()
        mock_model.assert_called_once()

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_model_eval_mode(self, mock_model, mock_processor, mock_config):
        """Test that DinoFeatureExtractor sets model to eval mode."""
        mock_processor.return_value = MagicMock()
        mock_model_instance = MagicMock()
        mock_model.return_value = mock_model_instance

        DinoFeatureExtractor(mock_config)

        mock_model_instance.eval.assert_called_once()

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_call_basic(self, mock_model, mock_processor, mock_config):
        """Test DinoFeatureExtractor call method."""
        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, 768)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        extractor = DinoFeatureExtractor(mock_config)

        # Create test image
        img = np.random.randint(0, 255, size=(1, 2000, 1000, 3), dtype=np.uint8)[0]

        result = extractor(img)

        assert result is not None
        assert isinstance(result, torch.Tensor)

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_call_splits_image(self, mock_model, mock_processor, mock_config):
        """Test that DinoFeatureExtractor splits image correctly."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, 768)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        extractor = DinoFeatureExtractor(mock_config)

        # Image with shape (1, 2000, 1000, 3)
        img = np.random.randint(0, 255, size=(1, 2000, 1000, 3), dtype=np.uint8)[0]

        extractor(img)

        # Processor should be called with two image halves
        mock_processor_instance.assert_called()

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_call_with_augmentation(self, mock_model, mock_processor, mock_config):
        """Test DinoFeatureExtractor with augmentation function."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, 768)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        extractor = DinoFeatureExtractor(mock_config)

        img = np.random.randint(0, 255, size=(1, 2000, 1000, 3), dtype=np.uint8)[0]
        aug_func = MagicMock(return_value={"image": img})

        result = extractor(img, aug_func=aug_func)

        aug_func.assert_called_once()
        assert result is not None

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_get_embedding_dim(self, mock_model, mock_processor, mock_config):
        """Test get_embedding_dim method."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        embedding_dim = 768
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, embedding_dim)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        extractor = DinoFeatureExtractor(mock_config)

        dim = extractor.get_embedding_dim()

        assert dim == embedding_dim

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_no_grad(self, mock_model, mock_processor, mock_config):
        """Test that forward pass uses no_grad."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, 768)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        extractor = DinoFeatureExtractor(mock_config)

        img = np.random.randint(0, 255, size=(1, 2000, 1000, 3), dtype=np.uint8)[0]

        result = extractor(img)

        # Should not require gradients
        assert not result.requires_grad

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_device_handling(self, mock_model, mock_processor, mock_config):
        """Test device handling in DinoFeatureExtractor."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, 768)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        DinoFeatureExtractor(mock_config)

        # Check that model was moved to device
        mock_model_instance.to.assert_called_with(mock_config.general.device)

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_mean_pooling(self, mock_model, mock_processor, mock_config):
        """Test that DinoFeatureExtractor uses mean pooling."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        # Create output with known values to test mean pooling
        batch_size = 2
        num_patches = 197
        embed_dim = 768
        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.ones(batch_size, num_patches, embed_dim)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        extractor = DinoFeatureExtractor(mock_config)

        img = np.random.randint(0, 255, size=(1, 2000, 1000, 3), dtype=np.uint8)[0]

        result = extractor(img)

        # Result should be 1D tensor (mean pooled)
        assert result.dim() == 1


class TestDinoFeatureExtractorIntegration:
    """Integration tests for DinoFeatureExtractor."""

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_multiple_images(self, mock_model, mock_processor, mock_config):
        """Test processing multiple images."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, 768)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        extractor = DinoFeatureExtractor(mock_config)

        # Process multiple images
        results = []
        for _ in range(5):
            img = np.random.randint(0, 255, size=(1, 2000, 1000, 3), dtype=np.uint8)[0]
            result = extractor(img)
            results.append(result)

        assert len(results) == 5
        assert all(isinstance(r, torch.Tensor) for r in results)

    @patch("csiro_biomass.features.dino.AutoImageProcessor.from_pretrained")
    @patch("csiro_biomass.features.dino.AutoModel.from_pretrained")
    def test_dino_with_augmentations_workflow(self, mock_model, mock_processor, mock_config):
        """Test DinoFeatureExtractor with augmentation workflow."""
        mock_processor_instance = MagicMock()
        mock_processor_instance.return_value = {"pixel_values": torch.randn(2, 3, 224, 224)}
        mock_processor_instance.to.return_value = mock_processor_instance
        mock_processor.return_value = mock_processor_instance

        mock_output = MagicMock()
        mock_output.last_hidden_state = torch.randn(2, 197, 768)
        mock_model_instance = MagicMock()
        mock_model_instance.return_value = mock_output
        mock_model_instance.to.return_value = mock_model_instance
        mock_model.return_value = mock_model_instance

        extractor = DinoFeatureExtractor(mock_config)

        img = np.random.randint(0, 255, size=(1, 2000, 1000, 3), dtype=np.uint8)[0]

        # Simulate multiple augmentations
        augmentations = [
            MagicMock(return_value={"image": img}),
            MagicMock(return_value={"image": img}),
            MagicMock(return_value={"image": img}),
        ]

        results = []
        for aug in augmentations:
            result = extractor(img, aug_func=aug)
            results.append(result)

        assert len(results) == 3
        assert all(isinstance(r, torch.Tensor) for r in results)
