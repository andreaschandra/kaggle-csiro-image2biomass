"""Integration tests for csiro_biomass.pipeline module."""

from unittest.mock import MagicMock, patch

import torch

from csiro_biomass.pipeline import Pipeline


class TestPipeline:
    """Tests for Pipeline class."""

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    def test_pipeline_initialization(
        self,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline initialization."""
        pipeline = Pipeline(mock_config, mock_logger)

        assert pipeline is not None
        assert pipeline.config == mock_config
        assert pipeline.logger == mock_logger

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    def test_pipeline_set_dataset(
        self,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline set_dataset method."""
        mock_extractor_instance = MagicMock()
        mock_extractor.return_value = mock_extractor_instance

        pipeline = Pipeline(mock_config, mock_logger)

        assert pipeline.feature_extractor is not None
        assert pipeline.dataset is not None
        mock_extractor.assert_called_once()
        mock_dataset.assert_called_once()

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    def test_pipeline_set_model(
        self,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline set_model method."""
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.get_embedding_dim.return_value = 768
        mock_extractor.return_value = mock_extractor_instance

        pipeline = Pipeline(mock_config, mock_logger)

        assert pipeline.model is not None
        mock_mlp.assert_called_once()

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    def test_pipeline_set_optimizer(
        self,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline set_optimizer method."""
        pipeline = Pipeline(mock_config, mock_logger)

        assert pipeline.optimizer is not None
        mock_optimizer.assert_called_once()

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    def test_pipeline_set_criterion(
        self,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline set_criterion method."""
        pipeline = Pipeline(mock_config, mock_logger)

        assert pipeline.criterion is not None
        mock_loss.assert_called_once()

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    def test_pipeline_set_epoch_metrics(
        self,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline set_epoch_metrics method."""
        pipeline = Pipeline(mock_config, mock_logger)

        assert pipeline.metrics is not None
        assert "train_loss" in pipeline.metrics
        assert "valid_loss" in pipeline.metrics
        assert "train_r2" in pipeline.metrics
        assert "valid_r2" in pipeline.metrics

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_train(
        self,
        mock_dataloader,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline train method."""
        # Setup mocks
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.randn(4, 3)
        mock_mlp.return_value = mock_model_instance

        mock_criterion_instance = MagicMock()
        mock_criterion_instance.return_value = torch.tensor(0.5)
        mock_loss.return_value = mock_criterion_instance

        # Mock dataloader to return one batch
        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        pipeline = Pipeline(mock_config, mock_logger)
        pipeline.train(fold=0)

        # Verify training was called
        mock_dataset_instance.set_fold.assert_called_once_with(0)
        mock_dataset_instance.set_split.assert_called_once_with("train")

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_valid(
        self,
        mock_dataloader,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline valid method."""
        # Setup mocks
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.randn(4, 4, 3)  # TTA output
        mock_mlp.return_value = mock_model_instance

        mock_criterion_instance = MagicMock()
        mock_criterion_instance.return_value = torch.tensor(0.5)
        mock_loss.return_value = mock_criterion_instance

        # Mock dataloader to return one batch
        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        pipeline = Pipeline(mock_config, mock_logger)
        pipeline.valid(fold=0)

        # Verify validation was called
        mock_dataset_instance.set_fold.assert_called()
        mock_dataset_instance.set_split.assert_called_with("valid")

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_epoch(
        self,
        mock_dataloader,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline epoch method."""
        # Setup mocks
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.randn(4, 3)
        mock_mlp.return_value = mock_model_instance

        mock_criterion_instance = MagicMock()
        mock_criterion_instance.return_value = torch.tensor(0.5)
        mock_loss.return_value = mock_criterion_instance

        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        pipeline = Pipeline(mock_config, mock_logger)
        pipeline.epoch(fold=0, i_epoch=1)

        # Check that metrics were updated
        assert len(pipeline.data_metrics) > 0

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_routine(
        self,
        mock_dataloader,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
    ):
        """Test Pipeline routine method."""
        # Setup mocks
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.randn(4, 3)
        mock_mlp.return_value = mock_model_instance

        mock_criterion_instance = MagicMock()
        mock_criterion_instance.return_value = torch.tensor(0.5)
        mock_loss.return_value = mock_criterion_instance

        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        mock_config.trainer.epoch = 2  # Small number for testing

        pipeline = Pipeline(mock_config, mock_logger)
        pipeline.routine(fold=0)

        # Check that multiple epochs were run
        assert len(pipeline.data_metrics) >= 2

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.MLP")
    @patch("csiro_biomass.pipeline.Optimizer")
    @patch("csiro_biomass.pipeline.WeightedHuberLoss")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_cross_validate(
        self,
        mock_dataloader,
        mock_loss,
        mock_optimizer,
        mock_mlp,
        mock_dataset,
        mock_extractor,
        mock_config,
        mock_logger,
        tmp_path,
    ):
        """Test Pipeline cross_validate method."""
        # Setup mocks
        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        mock_model_instance = MagicMock()
        mock_model_instance.return_value = torch.randn(4, 3)
        mock_model_instance.save = MagicMock()
        mock_mlp.return_value = mock_model_instance

        mock_criterion_instance = MagicMock()
        mock_criterion_instance.return_value = torch.tensor(0.5)
        mock_loss.return_value = mock_criterion_instance

        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        mock_config.dataset.kfolds = 2
        mock_config.general.kfolds = 2
        mock_config.trainer.epoch = 1
        mock_config.general.model_dir = str(tmp_path)

        pipeline = Pipeline(mock_config, mock_logger)
        pipeline.cross_validate()

        # Verify model save was called for each fold
        assert mock_model_instance.save.call_count >= 2


class TestPipelineIntegration:
    """Integration tests for Pipeline."""

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_full_workflow(
        self, mock_dataloader, mock_dataset, mock_extractor, mock_config, mock_logger, tmp_path
    ):
        """Test complete pipeline workflow."""
        # Setup realistic mocks
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.get_embedding_dim.return_value = 768
        mock_extractor.return_value = mock_extractor_instance

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        # Create data batches
        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        mock_config.dataset.kfolds = 2
        mock_config.general.kfolds = 2
        mock_config.trainer.epoch = 2
        mock_config.general.model_dir = str(tmp_path)

        pipeline = Pipeline(mock_config, mock_logger)

        # Run a single fold
        pipeline.set_epoch_metrics()
        pipeline.train(fold=0)
        pipeline.valid(fold=0)

        # Check metrics were computed
        assert pipeline.metrics["train_loss"] >= 0
        assert pipeline.metrics["valid_loss"] >= 0

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_metrics_tracking(
        self, mock_dataloader, mock_dataset, mock_extractor, mock_config, mock_logger
    ):
        """Test that pipeline tracks metrics correctly."""
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.get_embedding_dim.return_value = 768
        mock_extractor.return_value = mock_extractor_instance

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        mock_config.trainer.epoch = 3

        pipeline = Pipeline(mock_config, mock_logger)

        # Run routine
        pipeline.routine(fold=0)

        # Check that metrics were collected for each epoch
        assert len(pipeline.data_metrics) == 3
        for metrics in pipeline.data_metrics:
            assert "train_loss" in metrics
            assert "valid_loss" in metrics
            assert "train_r2" in metrics
            assert "valid_r2" in metrics

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_logging_calls(
        self, mock_dataloader, mock_dataset, mock_extractor, mock_config, mock_logger
    ):
        """Test that pipeline makes logging calls."""
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.get_embedding_dim.return_value = 768
        mock_extractor.return_value = mock_extractor_instance

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        mock_config.trainer.epoch = 1

        pipeline = Pipeline(mock_config, mock_logger)
        pipeline.epoch(fold=0, i_epoch=1)

        # Verify logger was called
        assert mock_logger.info.call_count > 0

    @patch("csiro_biomass.pipeline.DinoFeatureExtractor")
    @patch("csiro_biomass.pipeline.CSIRO")
    @patch("csiro_biomass.pipeline.DataLoader")
    def test_pipeline_device_handling(
        self, mock_dataloader, mock_dataset, mock_extractor, mock_config, mock_logger
    ):
        """Test that pipeline handles device correctly."""
        mock_extractor_instance = MagicMock()
        mock_extractor_instance.get_embedding_dim.return_value = 768
        mock_extractor.return_value = mock_extractor_instance

        mock_dataset_instance = MagicMock()
        mock_dataset.return_value = mock_dataset_instance

        device = torch.device("cpu")
        mock_config.general.device = device

        batch_x = torch.randn(4, 768)
        batch_y = torch.randn(4, 3)
        mock_dataloader.return_value = [(batch_x, batch_y)]

        pipeline = Pipeline(mock_config, mock_logger)

        # Model should be on the correct device
        assert all(p.device.type == "cpu" for p in pipeline.model.parameters())
