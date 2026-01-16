"""MLP Regressor model."""

import torch.nn as nn

from .base import BaseModel


class MLP(nn.Module, BaseModel):
    """MLP regressor model."""

    def __init__(self, config, emb_size):
        super().__init__()

        self.config = config
        self.model = nn.Sequential(
            nn.Linear(emb_size, 512),
            nn.LeakyReLU(),
            nn.Dropout2d(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Dropout2d(0.1),
            nn.Linear(256, 3),
            nn.ReLU(),
        )

    def forward(self, input_x):
        """Forward pass."""
        out = self.model(input_x)

        return out
