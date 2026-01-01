"""Finetune ConveNeXtv2 Regression."""

import torch.nn as nn

from .base import BaseModel


class ConvNeXtV2Regressor(nn.Module, BaseModel):
    def __init__(
        self,
        config,
        emb_size: int,
        drop_rate: float = 0.2,
    ):
        super().__init__()

        self.head = nn.Sequential(
            nn.LayerNorm(emb_size),
            nn.Dropout(drop_rate),
            nn.Linear(emb_size, 256),
            nn.GELU(),
            nn.Dropout(drop_rate / 2),
            nn.Linear(256, 3),
            nn.ReLU(),
        )

        self._init_head()

    def _init_head(
        self,
    ):
        for m in self.head.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        out = self.head(x)

        return out

    def get_params(self, lr: float, weight_decay: float):
        head_params = []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name:
                head_params.append(param)

        return [
            {"params": head_params, "lr": lr, "weight_decay": weight_decay},
        ]
