"""Loss Functions for CSIRO Biomass Competition"""

import torch
import torch.nn as nn


class WeightedMSELoss(nn.Module):
    """Weighted Mean Squared Error Loss."""

    def __init__(self):
        super().__init__()

        # Competition weights
        self.weights = torch.tensor(
            [0.1, 0.1, 0.1, 0.5, 0.2]
        )  # [Dry_Clover, Dry_Dead, Dry_Green, Dry_Total, GDM]

    def forward(self, pred, target):
        """Compute the weighted MSE loss."""
        # pred, target shape: [batch_size, 5]
        mse = (pred - target) ** 2
        weighted_mse = mse * self.weights.to(pred.device)

        return weighted_mse.mean()


class WeightedHuberLoss(nn.Module):
    """Weighted Huber Loss."""

    def __init__(self, delta=30, device="cpu"):
        super().__init__()
        self.delta = delta
        self.weights = torch.tensor([0.3, 0.3, 0.4]).to(device)

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

    def __init__(self, p=1.5):
        super().__init__()
        assert 1 < p < 2, "p must be between 1 and 2 for compound Poisson-Gamma"
        self.p = p

    def forward(self, y_pred, y_true):
        """Compute the Tweedie loss."""
        # Ensure predictions are positive
        y_pred = torch.clamp(y_pred, min=1e-8)

        # Tweedie deviance
        a = y_true * torch.pow(y_pred, 1 - self.p) / (1 - self.p)
        b = torch.pow(y_pred, 2 - self.p) / (2 - self.p)

        loss = -a + b
        return loss.mean()
