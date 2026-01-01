"""Pipeline module for training, validation, and testing."""

import os
import time

import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from csiro_biomass.dataset import CSIRO
from csiro_biomass.features import get_feature_extractor
from csiro_biomass.loss import WeightedHuberLoss
from csiro_biomass.metrics import compute_weighted_r2
from csiro_biomass.models import get_model_regressor
from csiro_biomass.optimizer import Optimizer
from csiro_biomass.utils.io import save_model


class Pipeline:
    """Main pipeline to run training, validation, and testing."""

    def __init__(self, config, logger, wandb_logger):
        self.config = config
        self.logger = logger
        self.wandb_logger = wandb_logger

        self.feature_extractor = None
        self.dataset = None
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.early_stopping = EarlyStopping(logger, patience=10, min_delta=0.09)

        self.set_dataset()
        self.set_model()
        self.set_optimizer()
        self.set_criterion()

        self.metrics = None

    def set_epoch_metrics(self, fold):
        """Set up metrics for each epoch."""
        self.metrics = {
            f"fold-{fold}/train_loss": 0,
            f"fold-{fold}/valid_loss": 0,
            f"fold-{fold}/train_r2": 0,
            f"fold-{fold}/valid_r2": 0,
            f"fold-{fold}/duration": 0,
            f"fold-{fold}/epoch": 0,
        }

    def set_dataset(self):
        """Set up the dataset and feature extractor."""
        if self.config.feature_extractor.is_enabled:
            self.logger.info("Load feature extractor")
            self.feature_extractor = get_feature_extractor(self.config)
            self.feature_extractor = self.feature_extractor.to(self.config.general.device)
            self.feature_extractor.eval()

        self.logger.info("Load Dataset")
        self.dataset = CSIRO(self.config, self.logger)

    def set_model(self):
        """Set up model regressor."""
        self.logger.info("Load MLP regressor model")
        torch.cuda.empty_cache()
        self.model = get_model_regressor(self.config, self.feature_extractor.get_embedding_dim())
        self.model = self.model.to(self.config.general.device)

    def set_optimizer(self):
        """Set up optimizer."""
        self.logger.info("Load optimizer")
        torch.cuda.empty_cache()
        self.optimizer = Optimizer(self.model, params=self.config.optimizer.params)
        scheduler_params = {
            "mode": "max",
            "factor": 0.5,
            "patience": 5,
            "threshold": 0.001,
            "min_lr": 0.00001,
        }
        self.scheduler = ReduceLROnPlateau(self.optimizer, **scheduler_params)

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
            self.dataset,
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            persistent_workers=(self.config.trainer.num_workers > 0),
        )
        self.model.train()
        for batch_index, (x, y) in enumerate(data_gen, 1):
            # Extract features for each sample in the batch
            x = x.to(self.config.general.device)
            y = y.to(self.config.general.device)

            # combine batches x total_tiles -> revert back to [batches, img_emb]
            batch_size, tiles, _, _, _ = x.shape
            x = x.reshape(-1, 3, 224, 224)
            x = self.feature_extractor(x, batch_size, tiles)

            self.model.zero_grad()

            out = self.model(x)
            loss = self.criterion(out, y)
            self.metrics[f"fold-{fold}/train_loss"] += (
                loss.item() - self.metrics[f"fold-{fold}/train_loss"]
            ) / batch_index

            r2 = compute_weighted_r2(out, y)
            self.metrics[f"fold-{fold}/train_r2"] += (
                r2 - self.metrics[f"fold-{fold}/train_r2"]
            ) / batch_index

            loss.backward()
            self.optimizer.step()

    def valid(self, fold):
        """Validate the model."""
        self.logger.info("Valid method called")
        self.logger.info(f"Fold: {fold}")
        self.dataset.set_fold(fold)
        self.dataset.set_split("valid")

        data_gen = DataLoader(
            self.dataset,
            batch_size=self.config.trainer.batch_size,
            shuffle=False,
            num_workers=self.config.trainer.num_workers,
            persistent_workers=(self.config.trainer.num_workers > 0),
        )
        self.model.eval()
        for batch_index, (x, y) in enumerate(data_gen, 1):
            # Extract features for each sample in the batch\
            x = x.to(self.config.general.device)
            y = y.to(self.config.general.device)

            # combine batches x total_tiles -> revert back to [batches, img_emb]
            batch_size, tiles, _, _, _ = x.shape
            x = x.reshape(-1, 3, 224, 224)
            x = self.feature_extractor(x, batch_size, tiles)

            with torch.no_grad():
                out = self.model(x)

            loss = self.criterion(out, y)
            self.metrics[f"fold-{fold}/valid_loss"] += (
                loss.item() - self.metrics[f"fold-{fold}/valid_loss"]
            ) / batch_index

            r2 = compute_weighted_r2(out, y)
            self.metrics[f"fold-{fold}/valid_r2"] += (
                r2 - self.metrics[f"fold-{fold}/valid_r2"]
            ) / batch_index

    def epoch(self, fold):
        """Run a single epoch of training and validation."""

        self.logger.info("Epoch method called")
        self.set_epoch_metrics(fold=fold)
        start = time.perf_counter()

        self.train(fold)
        self.valid(fold)
        self.scheduler.step(self.metrics[f"fold-{fold}/valid_r2"])

        duration = time.perf_counter() - start
        self.metrics[f"fold-{fold}/duration"] = duration

    def routine(self, fold):
        """Run the full training routine."""
        for i_epoch in tqdm(range(1, self.config.trainer.epoch + 1)):
            self.logger.info(f"Starting epoch {i_epoch}")
            self.epoch(fold)
            self.metrics[f"fold-{fold}/epoch"] = i_epoch
            self.logger.info(f"Finished epoch {i_epoch}")

            if self.early_stopping(self.metrics[f"fold-{fold}/valid_r2"], i_epoch):
                self.logger.info(f"Early stopping at epoch {i_epoch}")

                self.logger.info(
                    f"Epoch: {i_epoch} | "
                    f"lr: {self.scheduler.get_last_lr()} | "
                    f"Time: {self.metrics[f'fold-{fold}/duration']:.1f}s"
                )
                self.logger.info(
                    f"\tTrain loss: {self.metrics[f'fold-{fold}/train_loss']} | "
                    f" R2: {self.metrics[f'fold-{fold}/train_r2']}"
                )
                self.logger.info(
                    f"\tValid loss: {self.metrics[f'fold-{fold}/valid_loss']} | "
                    f" R2: {self.metrics[f'fold-{fold}/valid_r2']}"
                )
                break

            self.logger.info(
                f"Epoch: {i_epoch} | "
                f"lr: {self.scheduler.get_last_lr()} | "
                f"Time: {self.metrics[f'fold-{fold}/duration']:.1f}s"
            )
            self.logger.info(
                f"\tTrain loss: {self.metrics[f'fold-{fold}/train_loss']} | "
                f"R2: {self.metrics[f'fold-{fold}/train_r2']}"
            )
            self.logger.info(
                f"\tValid loss: {self.metrics[f'fold-{fold}/valid_loss']} | "
                f"R2: {self.metrics[f'fold-{fold}/valid_r2']}"
            )
            self.wandb_logger.log(self.metrics, commit=True)

    def cross_validate(self):
        """Run cross-validation."""

        self.logger.info("Cross-validation started")
        run_at = self.config.general.run_at
        model_dir = os.path.join(self.config.general.model_dir, run_at)
        for fold in range(self.config.dataset.kfolds):
            # reset all components
            self.set_model()
            self.set_optimizer()
            self.set_criterion()
            self.logger.info(f"Starting fold {fold}/{self.config.general.kfolds}")
            self.routine(fold)
            save_model(path=os.path.join(model_dir, f"{run_at}_fold-{fold}.pt"), model=self.model)

        return run_at, model_dir


class EarlyStopping:
    def __init__(self, logger=None, patience=5, min_delta=0.009):
        self.logger = logger
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = None
        self.counter = 0

    def __call__(self, val_score, epoch):
        if self.best_score is None:
            self.best_score = val_score
        elif (val_score < self.best_score + self.min_delta) and (epoch > 80):
            self.counter += 1
            self.logger.info("EarlyStopping Counter: {self.counter}")
            if self.counter >= self.patience:
                return True
        else:
            self.best_score = val_score
            self.counter = 0
        return False
