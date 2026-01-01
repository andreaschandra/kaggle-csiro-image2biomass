"""Weights & Biases (wandb) utilities for experiment tracking."""

from pathlib import Path
from typing import Any, Literal

import wandb


def init_wandb(
    entity: str,
    project: str,
    name: str = "default-run",
    config: dict[str, Any] | Any | None = None,
    tags: list[str] | None = ["training"],
    reinit: bool = False,
    mode: Literal["online", "offline", "disabled", "shared"] | None = "online",
) -> wandb.Run:
    """
    Initialize a wandb run for experiment tracking.

    Args:
        project: Project name
        name: Run name. If None, wandb generates a random name
        config: Configuration dictionary (hyperparameters, etc.)
        tags: List of tags for the run
        notes: Notes about the run
        entity: wandb entity (username or team name)
        reinit: Allow multiple wandb.init() calls in the same process
        resume: Resume a previous run. Can be True, False, "allow", "must", or "never"
        mode: Tracking mode - "online", "offline", or "disabled"

    Returns:
        wandb Run object

    Example:
        >>> from csiro_biomass.utils.wandb_utils import init_wandb
        >>> run = init_wandb(
        ...     project="csiro-biomass",
        ...     name="resnet50-experiment-1",
        ...     config={"learning_rate": 1e-4, "batch_size": 32},
        ...     tags=["resnet50", "baseline"]
        ... )
    """
    run = wandb.init(
        entity=entity,
        project=project,
        name=name,
        config=config.to_dict(),
        tags=tags,
        reinit=reinit,
        mode=mode,
    )

    for i in range(config.dataset.kfolds):
        run.define_metric(name=f"fold-{i}/epoch")
        run.define_metric(name=f"fold-{i}/train_loss", step_metric=f"fold-{i}/epoch")
        run.define_metric(name=f"fold-{i}/train_r2", step_metric=f"fold-{i}/epoch")
        run.define_metric(name=f"fold-{i}/valid_loss", step_metric=f"fold-{i}/epoch")
        run.define_metric(name=f"fold-{i}/valid_r2", step_metric=f"fold-{i}/epoch")
        run.define_metric(name=f"fold-{i}/duration", step_metric=f"fold-{i}/epoch")

    return run


def log_metrics(metrics: dict[str, Any], step: int | None = None, commit: bool = True) -> None:
    """
    Log metrics to wandb.

    Args:
        metrics: Dictionary of metric names and values
        step: Training step/epoch number
        commit: Whether to immediately save metrics to wandb server

    Example:
        >>> from csiro_biomass.utils.wandb_utils import log_metrics
        >>> log_metrics({
        ...     "train/loss": 0.45,
        ...     "train/accuracy": 0.89,
        ...     "val/loss": 0.52,
        ...     "val/accuracy": 0.85
        ... }, step=10)
    """
    wandb.log(metrics, step=step, commit=commit)


def log_model_checkpoint(
    checkpoint_path: str | Path,
    name: str | None = None,
    aliases: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> wandb.Artifact:
    """
    Log a model checkpoint as a wandb artifact.

    Args:
        checkpoint_path: Path to the checkpoint file or directory
        name: Artifact name. If None, uses checkpoint filename
        aliases: List of aliases for the artifact (e.g., ["latest", "best"])
        metadata: Additional metadata to store with the artifact

    Returns:
        wandb Artifact object

    Example:
        >>> from csiro_biomass.utils.wandb_utils import log_model_checkpoint
        >>> artifact = log_model_checkpoint(
        ...     checkpoint_path="./outputs/model_epoch_10.pth",
        ...     name="model-epoch-10",
        ...     aliases=["latest"],
        ...     metadata={"epoch": 10, "val_loss": 0.45}
        ... )
    """
    checkpoint_path = Path(checkpoint_path)

    if name is None:
        name = checkpoint_path.stem if checkpoint_path.is_file() else checkpoint_path.name

    artifact = wandb.Artifact(
        name=name,
        type="model",
        metadata=metadata,
    )

    if checkpoint_path.is_file():
        artifact.add_file(str(checkpoint_path))
    elif checkpoint_path.is_dir():
        artifact.add_dir(str(checkpoint_path))
    else:
        raise ValueError(f"Checkpoint path does not exist: {checkpoint_path}")

    wandb.log_artifact(artifact, aliases=aliases)
    return artifact


def watch_model(
    model: Any,
    log: str = "gradients",
    log_freq: int = 100,
    log_graph: bool = False,
) -> None:
    """
    Watch a PyTorch model to log gradients and parameters.

    Args:
        model: PyTorch model to watch
        log: What to log - "gradients", "parameters", "all", or None
        log_freq: Logging frequency (number of batches)
        log_graph: Whether to log the computational graph

    Example:
        >>> import torch.nn as nn
        >>> from csiro_biomass.utils.wandb_utils import watch_model
        >>> model = nn.Sequential(...)
        >>> watch_model(model, log="all", log_freq=100)
    """
    wandb.watch(model, log=log, log_freq=log_freq, log_graph=log_graph)


def finish_run(exit_code: int | None = None, quiet: bool = False) -> None:
    """
    Finish the current wandb run.

    Args:
        exit_code: Exit code for the run (0 for success, non-zero for failure)
        quiet: Whether to suppress output

    Example:
        >>> from csiro_biomass.utils.wandb_utils import finish_run
        >>> finish_run()
    """
    wandb.finish(exit_code=exit_code, quiet=quiet)


def log_config(config: dict[str, Any]) -> None:
    """
    Update the wandb config with additional parameters.

    Args:
        config: Configuration dictionary to log

    Example:
        >>> from csiro_biomass.utils.wandb_utils import log_config
        >>> log_config({"optimizer": "AdamW", "scheduler": "CosineAnnealingLR"})
    """
    wandb.config.update(config)


def log_table(
    name: str,
    data: list[list[Any]],
    columns: list[str],
) -> None:
    """
    Log a table to wandb.

    Args:
        name: Table name
        data: List of rows (each row is a list of values)
        columns: List of column names

    Example:
        >>> from csiro_biomass.utils.wandb_utils import log_table
        >>> log_table(
        ...     name="predictions",
        ...     data=[[1, 0.95, 1], [2, 0.87, 0]],
        ...     columns=["sample_id", "confidence", "prediction"]
        ... )
    """
    table = wandb.Table(data=data, columns=columns)
    wandb.log({name: table})


def log_image(
    name: str,
    image: Any,
    caption: str | None = None,
    step: int | None = None,
) -> None:
    """
    Log an image to wandb.

    Args:
        name: Image name/key
        image: Image to log (can be numpy array, PIL Image, or file path)
        caption: Optional caption for the image
        step: Training step/epoch number

    Example:
        >>> from csiro_biomass.utils.wandb_utils import log_image
        >>> import numpy as np
        >>> img = np.random.rand(224, 224, 3)
        >>> log_image("sample_image", img, caption="Random image", step=1)
    """
    wandb.log({name: wandb.Image(image, caption=caption)}, step=step)
