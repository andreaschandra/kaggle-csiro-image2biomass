"""Kaggle dataset download utilities using kagglehub."""

from pathlib import Path

import kagglehub


def authenticate_kaggle() -> None:
    """
    Authenticate with Kaggle using kagglehub.

    This will prompt for credentials if not already configured.
    Credentials can be set up by placing kaggle.json in ~/.kaggle/

    Example:
        >>> from csiro_biomass.utils.kaggle_utils import authenticate_kaggle
        >>> authenticate_kaggle()
    """
    kagglehub.login()


def download_kaggle_dataset(dataset: str) -> Path:
    """
    Download a Kaggle dataset using kagglehub.

    Note: kagglehub manages its own cache directory and downloads to a standard location.
    The downloaded data will be cached and reused on subsequent calls.

    Args:
        dataset: Dataset identifier in format 'owner/dataset-name'

    Returns:
        Path to the downloaded dataset directory in kagglehub's cache

    Example:
        >>> from csiro_biomass.utils.kaggle_utils import download_kaggle_dataset
        >>> dataset_path = download_kaggle_dataset("csiro/image2biomass")
        >>> print(f"Dataset available at: {dataset_path}")
    """
    # Download dataset using kagglehub
    download_path = kagglehub.dataset_download(dataset)

    print(f"Dataset '{dataset}' available at {download_path}")
    return Path(download_path)


def download_kaggle_competition_data(competition: str) -> Path:
    """
    Download Kaggle competition data using kagglehub.

    Note: kagglehub manages its own cache directory and downloads to a standard location.
    The downloaded data will be cached and reused on subsequent calls.

    Args:
        competition: Competition name (e.g., 'csiro-image2biomass')

    Returns:
        Path to the downloaded competition data directory in kagglehub's cache

    Example:
        >>> from csiro_biomass.utils.kaggle_utils import download_kaggle_competition_data
        >>> data_path = download_kaggle_competition_data("csiro-image2biomass")
        >>> print(f"Competition data available at: {data_path}")
    """
    # Download competition data using kagglehub
    download_path = kagglehub.competition_download(competition)

    print(f"Competition '{competition}' data available at {download_path}")
    return Path(download_path)
