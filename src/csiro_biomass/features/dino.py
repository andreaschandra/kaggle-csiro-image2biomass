"""DINO feature extractor."""

import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel

from csiro_biomass.features.base import BaseFeatureExtractor


class DinoFeatureExtractor(nn.Module, BaseFeatureExtractor):
    """DINO feature extractor."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(
            config.feature_extractor.pretrained_name, device=config.general.device
        )
        self.model = AutoModel.from_pretrained(config.feature_extractor.pretrained_name)
        self.model = self.model.to(config.general.device)
        self.model.eval()

    def forward(self, img, batch_size, num_tiles):
        with torch.no_grad():
            emb_tiles = self.model(**img)

        emb_tiles = emb_tiles.last_hidden_state[:, 1:, :]

        _, patch_size, emb_size = emb_tiles.shape
        emb_tiles = emb_tiles.reshape(batch_size, num_tiles, patch_size, emb_size)

        emb_mean = emb_tiles.mean(dim=[1, 2])

        return emb_mean

    def get_embedding_dim(self):
        """Get embedding dimension."""
        img_rand = torch.randint(low=0, high=255, size=(2000, 1000, 3))
        x = self.processor(img_rand, return_tensors="pt")
        x = x.to(self.config.general.device)
        out = self(x, batch_size=1, num_tiles=1)
        # [batch_size, emb_size]
        return out.shape[1]
