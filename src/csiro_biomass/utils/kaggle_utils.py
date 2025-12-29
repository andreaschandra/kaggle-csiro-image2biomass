"""Kaggle dataset download utilities using kagglehub."""

import os
from pathlib import Path

import kagglehub
from kagglehub.config import set_kaggle_credentials


def authenticate_kaggle() -> None:
    """
    Authenticate with Kaggle using kagglehub.

    This will prompt for credentials if not already configured.
    Credentials can be set up by placing kaggle.json in ~/.kaggle/

    Example:
        >>> from csiro_biomass.utils.kaggle_utils import authenticate_kaggle
        >>> authenticate_kaggle()
    """
    set_kaggle_credentials(username=os.getenv("KAGGLE_USERNAME"), api_key=os.getenv("KAGGLE_KEY"))


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


def upload_model_dir_to_kaggle(model_name: str, version: str, model_dir) -> None:
    """
    Upload a model directory to Kaggle using kagglehub.

    Args:
        model_handle: Model identifier in format 'owner/model-name'
        model_dir: Path to the local model directory to upload

    Example:
        >>> from csiro_biomass.utils.kaggle_utils import upload_model_dir_to_kaggle
        >>> from pathlib import Path
        >>> upload_model_dir_to_kaggle("csiro/my-model", Path("/path/to/my/model"))
    """
    model_handle = os.path.join("andreaschandra", model_name, "pyTorch", version)
    kagglehub.model_upload(model_handle, model_dir)
