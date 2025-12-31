"""Models module."""

from csiro_biomass.models.base import BaseModel
from csiro_biomass.models.convnextv2 import ConvNeXtV2Regressor
from csiro_biomass.models.mlp import MLP

__all__ = ["MLP", "ConvNeXtV2Regressor"]


def get_model_regressor(config, emb_size):
    """Get available model classes."""

    for cls in BaseModel.__subclasses__():
        if config.regressor.model == cls.__name__:
            print(cls.__name__)
            return cls(config, emb_size=emb_size)

    raise ValueError(f"Model name not found, given: {config.regressor.model}")
