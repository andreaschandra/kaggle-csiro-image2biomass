"""Loss Functions for CSIRO Biomass Competition"""

import torch
import torch.nn as nn


class BaseLoss(nn.Module):
    def __init__(self):
        super().__init__()


class WeightedMSELoss(BaseLoss):
    """Weighted Mean Squared Error Loss."""

    def __init__(self, config):
        super().__init__()

        # Competition weights
        self.weights = torch.tensor([0.33, 0.33, 0.34]).to(config.general.device)

    def forward(self, pred, target):
        """Compute the weighted MSE loss."""
        mse = (pred - target) ** 2
        weighted_mse = mse * self.weights.to(pred.device)

        return weighted_mse.sum(dim=1).mean()


class WeightedHuberLoss(BaseLoss):
    """Weighted Huber Loss."""

    def __init__(self, config, delta=30):
        super().__init__()
        self.delta = delta
        self.weights = torch.tensor([0.3, 0.3, 0.4]).to(config.general.device)

    def forward(self, pred, target):
        """Compute the weighted Huber loss."""
        error = pred - target
        is_small = torch.abs(error) <= self.delta
        small_loss = 0.5 * error**2
        large_loss = self.delta * torch.abs(error) - 0.5 * self.delta**2

        # Weight by target magnitude
        loss = torch.where(is_small, small_loss, large_loss)
        weighted_loss = loss * self.weights

        return weighted_loss.mean()


class TweedieLoss(nn.Module):
    """Tweedie Loss for compound Poisson-Gamma distribution."""

    def __init__(
        self,
        config,
        p=1.5,
    ):
        super().__init__()
        assert 1 < p < 2, "p must be between 1 and 2 for compound Poisson-Gamma"
        self.config = config
        self.p = p

    def forward(self, pred, target):
        """Compute the Tweedie loss."""
        # Ensure predictions are positive
        y_pred = torch.clamp(pred, min=1e-8)

        # Tweedie deviance
        a = target * torch.pow(y_pred, 1 - self.p) / (1 - self.p)
        b = torch.pow(y_pred, 2 - self.p) / (2 - self.p)

        loss = -a + b
        return loss.mean()


def get_loss_function(config):
    for cls in BaseLoss.__subclasses__():
        if cls.__name__ == config.loss.type:
            return cls(config=config)

    raise ValueError(f"Loss function {config.loss.type} is not available.")
