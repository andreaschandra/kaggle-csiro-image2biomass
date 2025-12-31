"""Features module."""

from csiro_biomass.features.base import BaseFeatureExtractor
from csiro_biomass.features.convnextv2 import ConvNeXtV2FeatureExtractor
from csiro_biomass.features.dino import DinoFeatureExtractor

__all__ = ["DinoFeatureExtractor", "ConvNeXtV2FeatureExtractor"]


def get_feature_extractor(config, kwargs={}):
    """Get available model classes."""

    for cls in BaseFeatureExtractor.__subclasses__():
        if config.feature_extractor.model == cls.__name__:
            print(cls.__name__)
            return cls(config, **kwargs)

    raise ValueError(f"Model name not found, given: {config.feature_extractor.model}")
