"""Main module for CSIRO Biomass project."""

import argparse

from dotenv import load_dotenv

from csiro_biomass.config import Config
from csiro_biomass.pipeline import Pipeline
from csiro_biomass.utils.logger import setup_logger
from csiro_biomass.utils.wandb_utils import init_wandb


def main(args_main):
    """Main function to run the CSIRO Biomass project."""
    logger = setup_logger()
    config = Config.load_from_file(args_main.config)

    wandb_run = init_wandb(
        entity=config.wandb.entity,
        project=config.wandb.project,
        name=config.general.run_at,
        config=config,
    )

    logger.info("Starting CSIRO Biomass project...")
    logger.info("Load config")
    logger.info("Setup pipeline")
    pl = Pipeline(config, logger, wandb_run)

    logger.info("Run cross validation")
    run_at, model_dir = pl.cross_validate()

    # logger.info("Upload model to Huggingface Hub")
    # commit_hash = get_current_commit_hash()
    # upload_model_folder_to_hf(
    #     repo_id=config.huggingface.repo_id,
    #     folder_path=model_dir,
    #     commit_message=f"model {run_at} {commit_hash}",
    #     tag=run_at,
    # )
    # upload_model_dir_to_kaggle(
    #     model_name=config.general.competition,
    #     version=run_at,
    #     model_dir=model_dir,
    # )
    # finish_run()


def arg_parser():
    """Parse command line arguments."""

    parser = argparse.ArgumentParser(description="CSIRO Biomass Project")
    parser.add_argument(
        "--config",
        type=str,
        required=False,
        default="configs/train.yaml",
        help="Path to the configuration file.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = arg_parser()
    load_dotenv()
    main(args)
