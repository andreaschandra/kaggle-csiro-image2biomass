"""Optimizer model."""

import torch.optim as optim


class Optimizer:
    """Optimizer."""

    def __init__(self, model, params=None):

        if params is None:
            params = {"lr": 1e-3}

        self.optimizer = optim.AdamW(model.parameters(), **params)

    def step(self):
        """Gradient step."""
        self.optimizer.step()
