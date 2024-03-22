import pyprojroot
import sys

root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from pprint import pprint
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary,
)


from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation

import wandb


@dataclass
class ESDConfig:
    """
    IMPORTANT: This class is used to define the configuration for the experiment
    Please make sure to use the correct types and default values for the parameters
    and that the path for processed_dir contain the tiles you would like
    """

    processed_dir: str | os.PathLike = root / "data/processed/4x4"
    raw_dir: str | os.PathLike = root / "data/raw/Train"
    selected_bands: None | dict[str, list[str]] = None
    model_type: str = "SegmentationCNN"
    tile_size_gt: int = 4
    batch_size: int = 8
    max_epochs: int = 2
    seed: int = 12378921
    learning_rate: float = 1e-3
    num_workers: int = 11
    accelerator: str = "cpu"
    devices: int = 1
    in_channels: int = 99
    out_channels: int = 4
    depth: int = 2
    n_encoders: int = 2
    embedding_size: int = 64
    pool_sizes: tuple[int] = (5, 5, 2)
    kernel_size: int = 3
    scale_factor: int = 50
    wandb_run_name: str | None = None


def train(options: ESDConfig):
    """
    Prepares datamodule and model, then runs the training loop

    Inputs:
        options: ESDConfig
            options for the experiment
    """
    # Initialize the weights and biases logger
    pprint(f"{options=}")
    wandb.init()
    wandb_logger = pl.loggers.WandbLogger(
        project="testing_models", entity="afropixel", name=options.wandb_run_name
    )

    # initiate the ESDDatamodule
    # use the options object to initiate the datamodule correctly
    # make sure to prepare_data in case the data has not been preprocessed
    data_module = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        selected_bands=options.selected_bands,
    )

    data_module.prepare_data()

    # create a dictionary with the parameters to pass to the models
    model_params = {
        "depth": options.depth,
        "n_encoders": options.n_encoders,
        "embedding_size": options.embedding_size,
        "pool_sizes": options.pool_sizes,
        "kernel_size": options.kernel_size,
        "scale_factor": options.scale_factor,
    }

    # initialize the ESDSegmentation module
    esd_segmentation = ESDSegmentation(
        options.model_type,
        options.in_channels,
        options.out_channels,
        options.learning_rate,
        model_params,
    )

    # Use the following callbacks, they're provided for you,
    # but you may change some of the settings
    # ModelCheckpoint: saves intermediate results for the neural network
    # in case it crashes
    # LearningRateMonitor: logs the current learning rate on weights and biases
    # RichProgressBar: nicer looking progress bar (requires the rich package)
    # RichModelSummary: shows a summary of the model before training (requires rich)
    callbacks = [
        ModelCheckpoint(
            dirpath=root / "models" / options.model_type,
            filename="{epoch}-{val_loss:.2f}-{other_metric:.2f}",
            save_top_k=0,
            save_last=True,
            verbose=True,
            monitor="val_loss",
            mode="min",
            every_n_train_steps=1000,
        ),
        LearningRateMonitor(),
        # RichProgressBar(),
        RichModelSummary(max_depth=3),
    ]


    # make sure to use the options object to load it with the correct options
    # run trainer.fit after initializing the trainer
    trainer = pl.Trainer(
        logger=wandb_logger,
        max_epochs=options.max_epochs,
        accelerator=options.accelerator,
        devices=options.devices,
        callbacks=callbacks,
        log_every_n_steps=1
    )
    trainer.fit(
        model=esd_segmentation, datamodule=data_module
    )


if __name__ == "__main__":
    # load dataclass arguments from yml file

    config = ESDConfig()
    parser = ArgumentParser()

    parser.add_argument(
        "--model_type",
        type=str,
        help="The model to initialize.",
        default=config.model_type,
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        help="The learning rate for training model",
        default=config.learning_rate,
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        help="Number of epochs to train for.",
        default=config.max_epochs,
    )
    parser.add_argument(
        "--raw_dir", type=str, default=config.raw_dir, help="Path to raw directory"
    )
    parser.add_argument(
        "-p", "--processed_dir", type=str, default=config.processed_dir, help="."
    )

    parser.add_argument(
        "--in_channels",
        type=int,
        default=config.in_channels,
        help="Number of input channels",
    )
    parser.add_argument(
        "--out_channels",
        type=int,
        default=config.out_channels,
        help="Number of output channels",
    )
    parser.add_argument(
        "--depth",
        type=int,
        help="Depth of the encoders (CNN only)",
        default=config.depth,
    )
    parser.add_argument(
        "--n_encoders",
        type=int,
        help="Number of encoders (Unet only)",
        default=config.n_encoders,
    )
    parser.add_argument(
        "--embedding_size",
        type=int,
        help="Embedding size of the neural network (CNN/Unet)",
        default=config.embedding_size,
    )
    parser.add_argument(
        "--pool_sizes",
        help="A comma separated list of pool_sizes (CNN only)",
        type=str,
        default=config.pool_sizes,
    )
    parser.add_argument(
        "--kernel_size",
        help="Kernel size of the convolutions",
        type=int,
        default=config.kernel_size,
    )
    parser.add_argument(
        "--scale_factor",
        help="Scale factor between the labels and the image (Unet and Transfer Resnet)",
        type=int,
        default=config.scale_factor,
    )

    parse_args = parser.parse_args()
    train_config = ESDConfig(**parse_args.__dict__)
    # modify your selected_bands here
    train_config.selected_bands = {"viirs_maxproj": ["0"], "gt": ["0"]}
    train(train_config)
