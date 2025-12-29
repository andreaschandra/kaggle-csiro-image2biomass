"""MLP Regressor model."""


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
            nn.Linear(512, 256),
            nn.LeakyReLU(),
            nn.Linear(256, 3),
            nn.ReLU(),
        )

    def forward(self, input_x):
        """Forward pass."""
        out = self.model(input_x)

        return out
