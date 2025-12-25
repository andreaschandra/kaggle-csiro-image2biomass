"""Main module for CSIRO Biomass project."""

import argparse

from csiro_biomass.config import Config
from csiro_biomass.pipeline import Pipeline
from csiro_biomass.utils.logger import setup_logger


def main(args_main):
    """Main function to run the CSIRO Biomass project."""
    logger = setup_logger()
    logger.info("Starting CSIRO Biomass project...")
    logger.info("Load config")
    config = Config.load_from_file(args_main.config)
    logger.info("Setup pipeline")
    pl = Pipeline(config, logger)
    pl.cross_validate()


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
    main(args)
