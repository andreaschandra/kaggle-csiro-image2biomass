"""ConvNeXtV2 Image Embeddings."""

import timm
import torch
import torch.nn as nn

from csiro_biomass.features.base import BaseFeatureExtractor


class ConvNeXtV2FeatureExtractor(nn.Module, BaseFeatureExtractor):
    def __init__(
        self,
        config,
        drop_rate: float = 0.3,
    ):
        super().__init__()

        self.config = config
        data_config = timm.data.resolve_model_data_config(config.feature_extractor.pretrained_name)
        self.transform = timm.data.create_transform(**data_config, is_training=False)
        self.backbone = timm.create_model(
            model_name=config.feature_extractor.pretrained_name,
            pretrained=True,
            drop_rate=drop_rate,
            num_classes=0,
        )
        self.backbone = self.backbone.to(config.general.device)
        self.backbone = self.backbone.eval()

        self.freeze_backbone()

    def forward(self, x):
        emb_tiles = []
        for tile in x:
            x = self.transform(tile)
            x = x.unsqueeze(0)
            x = x.to(self.config.general.device)

            with torch.no_grad():
                out = self.backbone(x)
                emb_tiles.append(out)

        emb_tiles = torch.cat(emb_tiles, dim=0)
        emb_mean = emb_tiles.mean(dim=0)
        return emb_mean

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self, num_stages: int = 2):
        for param in self.backbone.parameters():
            param.requires_grad = False

        stages = list(self.backbone.stages)
        for stage in stages[-num_stages:]:
            for param in stage.parameters():
                param.requires_grad = True

        if hasattr(self.backbone, "norm"):
            for param in self.backbone.norm.parameters():
                param.requires_grad = True

    def get_param_groups(self, backbone_lr: float, head_lr: float, weight_decay: float):
        backbone_params = []
        head_params = []

        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if "head" in name:
                head_params.append(param)
            else:
                backbone_params.append(param)

        return [
            {"params": backbone_params, "lr": backbone_lr, "weight_decay": weight_decay},
            {"params": head_params, "lr": head_lr, "weight_decay": weight_decay},
        ]

    def get_embedding_dim(self):
        """get embedding size."""
        return self.backbone.num_features
