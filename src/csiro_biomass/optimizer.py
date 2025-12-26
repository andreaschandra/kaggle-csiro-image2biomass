"""Optimizer model."""

import torch.optim as optim


class Optimizer(optim.AdamW):
    """Optimizer."""

    def __init__(self, model, params=None):
        if params is None:
            params = {"lr": 1e-3}

        super().__init__(model.parameters(), **params)
