"""IO utilities."""

import os

import pandas as pd
import torch
import yaml


def read_yaml(path):
    """Read YAML file."""

    with open(path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    return config_data


def read_csv(path):
    """Read CSV file into DataFrame."""
    d_data = pd.read_csv(path)
    return d_data


def write_csv(path, df):
    """Write DataFrame to CSV file."""
    df.to_csv(path, index=False)


def save_model(path, model):
    """Save model to file."""

    if os.path.exists(os.path.dirname(path)) is False:
        os.makedirs(os.path.dirname(path))

    torch.save(model.model.state_dict(), path)
