"""IO utilities."""

import pandas as pd
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
