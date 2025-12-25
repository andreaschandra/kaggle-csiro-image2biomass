"""DINO feature extractor."""

import torch
from transformers import AutoImageProcessor, AutoModel


class DinoFeatureExtractor:
    """DINO feature extractor."""

    def __init__(self, config):
        self.config = config
        self.processor = AutoImageProcessor.from_pretrained(
            config.feature_extractor.model, device=config.general.device
        )
        self.model = AutoModel.from_pretrained(config.feature_extractor.model)
        self.model = self.model.to(config.general.device)
        self.model.eval()

    def __call__(self, img, aug_func=None):
        if aug_func:
            img = aug_func(image=img)["image"]

        img1 = img[:, :1000]
        img2 = img[:, 1000:]
        inputs = self.processor(images=[img1, img2], return_tensors="pt")
        inputs = inputs.to(self.config.general.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # Multi-scale feature extraction
        patch_tokens = outputs.last_hidden_state[:, 1:, :]  # Patch tokens

        mean = patch_tokens.mean(dim=[0, 1])

        return mean

    def get_embedding_dim(self):
        """Get embedding dimension."""
        img_rand = torch.randint(low=0, high=255, size=(1, 2000, 1000, 3))
        out = self(img_rand)
        return out.shape[0]
