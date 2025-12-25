"""HuggingFace Hub utilities for model checkpoint management."""

from pathlib import Path

from huggingface_hub import create_repo, upload_file, upload_folder


def upload_checkpoint_to_hf(
    checkpoint_path: str | Path,
    repo_id: str = None,
    path_in_repo: str | None = None,
    commit_message: str | None = None,
    create_pr: bool = False,
    token: str | None = None,
) -> str:
    """
    Upload a model checkpoint file to HuggingFace Hub.

    Args:
        checkpoint_path: Local path to the checkpoint file
        repo_id: Repository ID on HuggingFace Hub (e.g., 'username/model-name')
        path_in_repo: Path where the file should be stored in the repo.
            If None, uses the filename from checkpoint_path
        commit_message: Custom commit message. If None, auto-generated
        create_pr: Whether to create a pull request instead of committing directly
        token: HuggingFace API token. If None, uses token from HF_TOKEN env var or cached token

    Returns:
        URL of the uploaded file

    Example:
        >>> from csiro_biomass.utils.hf_utils import upload_checkpoint_to_hf
        >>> url = upload_checkpoint_to_hf(
        ...     checkpoint_path="./outputs/model_epoch_10.pth",
        ...     repo_id="username/csiro-biomass-model",
        ...     commit_message="Add checkpoint from epoch 10"
        ... )
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    if path_in_repo is None:
        path_in_repo = checkpoint_path.name

    if commit_message is None:
        commit_message = f"Upload {checkpoint_path.name}"

    # Upload file
    url = upload_file(
        path_or_fileobj=str(checkpoint_path),
        path_in_repo=path_in_repo,
        repo_id=repo_id,
        commit_message=commit_message,
        create_pr=create_pr,
        token=token,
    )

    print(f"Checkpoint uploaded to: {url}")
    return url


def upload_model_folder_to_hf(
    folder_path: str | Path,
    repo_id: str,
    commit_message: str | None = None,
    create_pr: bool = False,
    token: str | None = None,
    ignore_patterns: list[str] | None = None,
) -> str:
    """
    Upload an entire model folder to HuggingFace Hub.

    Args:
        folder_path: Local path to the model folder
        repo_id: Repository ID on HuggingFace Hub (e.g., 'username/model-name')
        commit_message: Custom commit message. If None, auto-generated
        create_pr: Whether to create a pull request instead of committing directly
        token: HuggingFace API token. If None, uses token from HF_TOKEN env var or cached token
        ignore_patterns: List of patterns to ignore (e.g., ['*.git*', '__pycache__'])

    Returns:
        URL of the uploaded folder

    Example:
        >>> from csiro_biomass.utils.hf_utils import upload_model_folder_to_hf
        >>> url = upload_model_folder_to_hf(
        ...     folder_path="./outputs/final_model",
        ...     repo_id="username/csiro-biomass-model",
        ...     ignore_patterns=["*.log", "wandb/"]
        ... )
    """
    folder_path = Path(folder_path)

    if not folder_path.exists():
        raise FileNotFoundError(f"Folder not found: {folder_path}")

    if not folder_path.is_dir():
        raise ValueError(f"Path is not a directory: {folder_path}")

    if commit_message is None:
        commit_message = f"Upload model from {folder_path.name}"

    # Upload folder
    url = upload_folder(
        folder_path=str(folder_path),
        repo_id=repo_id,
        commit_message=commit_message,
        create_pr=create_pr,
        token=token,
        ignore_patterns=ignore_patterns,
    )

    print(f"Model folder uploaded to: {url}")
    return url


def create_hf_model_repo(
    repo_id: str,
    private: bool = False,
    token: str | None = None,
    exist_ok: bool = True,
) -> str:
    """
    Create a new model repository on HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., 'username/model-name')
        private: Whether to make the repository private
        token: HuggingFace API token. If None, uses token from HF_TOKEN env var or cached token
        exist_ok: If True, do not raise error if repo already exists

    Returns:
        URL of the created repository

    Example:
        >>> from csiro_biomass.utils.hf_utils import create_hf_model_repo
        >>> repo_url = create_hf_model_repo(
        ...     repo_id="username/csiro-biomass-model",
        ...     private=False
        ... )
    """
    url = create_repo(
        repo_id=repo_id,
        private=private,
        token=token,
        exist_ok=exist_ok,
        repo_type="model",
    )

    print(f"Repository created: {url}")
    return url


def get_checkpoint_url(repo_id: str, filename: str) -> str:
    """
    Get the direct URL to a checkpoint file on HuggingFace Hub.

    Args:
        repo_id: Repository ID (e.g., 'username/model-name')
        filename: Name of the checkpoint file

    Returns:
        Direct URL to the checkpoint file

    Example:
        >>> from csiro_biomass.utils.hf_utils import get_checkpoint_url
        >>> url = get_checkpoint_url(
        ...     repo_id="username/csiro-biomass-model",
        ...     filename="model_epoch_10.pth"
        ... )
    """
    return f"https://huggingface.co/{repo_id}/resolve/main/{filename}"
