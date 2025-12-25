"""Pipeline module for training, validation, and testing."""

import time
from datetime import datetime

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from csiro_biomass.dataset import CSIRO
from csiro_biomass.features.dino import DinoFeatureExtractor
from csiro_biomass.loss import WeightedHuberLoss
from csiro_biomass.metrics import compute_weighted_r2
from csiro_biomass.models.mlp import MLP
from csiro_biomass.optimizer import Optimizer


class Pipeline:
    """Main pipeline to run training, validation, and testing."""

    def __init__(self, config, logger):
        self.config = config
        self.logger = logger

        self.feature_extractor = None
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.criterion = None

        self.set_dataset()
        self.set_model()
        self.set_optimizer()
        self.set_criterion()

        self.metrics = None
        self.set_epoch_metrics()
        self.data_metrics = []

    def set_epoch_metrics(self):
        """Set up metrics for each epoch."""
        self.metrics = {
            "train_loss": 0,
            "valid_loss": 0,
            "train_r2": 0,
            "valid_r2": 0,
        }

    def set_dataset(self):
        """Set up the dataset and feature extractor."""
        self.logger.info("Load feature extractor")
        self.feature_extractor = DinoFeatureExtractor(self.config)
        self.logger.info("Load Dataset")
        self.dataset = CSIRO(self.config, self.feature_extractor)

    def set_model(self):
        """Set up model regressor."""
        self.logger.info("Load MLP regressor model")
        self.model = MLP(emb_size=self.feature_extractor.get_embedding_dim()).to(
            self.config.general.device
        )

    def set_optimizer(self):
        """Set up optimizer."""
        self.logger.info("Load optimizer")
        self.optimizer = Optimizer(self.model, params=self.config.optimizer.params)

    def set_criterion(self):
        """Set up criterion."""
        self.logger.info("Load criterion")
        self.criterion = WeightedHuberLoss(delta=3.0, device=self.config.general.device)

    def train(self, fold):
        """Train the model."""
        self.logger.info("Train method called")
        self.logger.info(f"Fold: {fold}")
        self.model.zero_grad()

        self.dataset.set_fold(fold)
        self.dataset.set_split("train")
        data_gen = DataLoader(
            self.dataset, batch_size=self.config.trainer.batch_size, shuffle=False
        )
        self.model.train()
        for batch_index, (x, y) in enumerate(data_gen, 1):
            x = x.to(self.config.general.device)
            y = y.to(self.config.general.device)

            self.model.zero_grad()

            out = self.model(x)
            loss = self.criterion(out, y)
            self.metrics["train_loss"] += (loss.item() - self.metrics["train_loss"]) / batch_index

            r2 = compute_weighted_r2(out, y)
            self.metrics["train_r2"] += (r2 - self.metrics["train_r2"]) / batch_index

            loss.backward()
            self.optimizer.step()
            break

        return

    def valid(self, fold):
        """Validate the model."""
        self.logger.info("Valid method called")
        self.logger.info(f"Fold: {fold}")
        self.dataset.set_fold(fold)
        self.dataset.set_split("valid")
        data_gen = DataLoader(
            self.dataset, batch_size=self.config.trainer.batch_size, shuffle=False
        )
        self.model.eval()
        for batch_index, (x, y) in enumerate(data_gen, 1):
            x = x.to(self.config.general.device)
            y = y.to(self.config.general.device)

            with torch.no_grad():
                out = self.model(x)
                out = out.mean(dim=1)

            loss = self.criterion(out, y)
            self.metrics["valid_loss"] += (loss.item() - self.metrics["valid_loss"]) / batch_index

            r2 = compute_weighted_r2(out, y)
            self.metrics["valid_r2"] += (r2 - self.metrics["valid_r2"]) / batch_index
            break

    def epoch(self, fold, i_epoch):
        """Run a single epoch of training and validation."""

        self.logger.info("Epoch method called")
        start = time.perf_counter()
        self.set_epoch_metrics()
        self.train(fold)
        self.valid(fold)
        duration = time.perf_counter() - start
        self.data_metrics.append(self.metrics.copy())

        self.logger.info(f"Epoch: {i_epoch} | Time: {duration:.1f}s")
        self.logger.info(
            f"\t Train loss: {self.metrics['train_loss']} | R2: {self.metrics['train_r2']}"
        )
        self.logger.info(
            f"\t Valid loss: {self.metrics['valid_loss']} | R2: {self.metrics['valid_r2']}"
        )

    def routine(self, fold):
        """Run the full training routine."""
        for i_epoch in tqdm(range(1, self.config.trainer.epoch + 1)):
            self.logger.info(f"Starting epoch {i_epoch}")
            self.epoch(fold, i_epoch)
            self.logger.info(f"Finished epoch {i_epoch}")

    def cross_validate(self):
        """Run cross-validation."""

        self.logger.info("Cross-validation started")
        ### reset all components
        self.set_dataset()
        self.set_model()
        self.set_optimizer()
        self.set_criterion()
        run_at = datetime.now().strftime("%Y%m%d-%H%M%S")
        for fold in range(self.config.dataset.kfolds):
            self.logger.info(f"Starting fold {fold + 1}/{self.config.general.kfolds}")
            self.routine(fold)
            self.model.save(
                path=f"{self.config.general.model_dir}/{run_at}/{run_at}_fold-{fold}.pt"
            )
