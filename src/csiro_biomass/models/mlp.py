"""MLP Regressor model."""

import os

import torch
import torch.nn as nn


class MLP(nn.Module):
    """MLP regressor model."""

    def __init__(self, emb_size):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(emb_size, 1024),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 3),
            nn.ReLU(),
        )

        self.model.apply(self.weights_init)

    def forward(self, input_x):
        """Forward pass."""
        out = self.model(input_x)

        return out

    def save(self, path):
        """Save model to path."""
        if os.path.exists(os.path.dirname(path)) is False:
            os.makedirs(os.path.dirname(path))
        torch.save(self.model.state_dict(), path)

    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight.data)
