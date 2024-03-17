import pyprojroot
import sys
import os
root = pyprojroot.here()
sys.path.append(str(root))
import pytorch_lightning as pl
from argparse import ArgumentParser
import os
from typing import List
from dataclasses import dataclass
from pathlib import Path

from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    RichProgressBar,
    RichModelSummary
)

from src.esd_data.datamodule import ESDDataModule
from src.models.supervised.satellite_module import ESDSegmentation
from src.preprocessing.subtile_esd_hw02 import Subtile
from src.visualization.restitch_plot import (
    restitch_eval,
    restitch_and_plot
)
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt 
import matplotlib
import tifffile
@dataclass
class EvalConfig:
    processed_dir: str | os.PathLike = root / 'data/processed/4x4'
    raw_dir: str | os.PathLike = root / 'data/raw/Train'
    #results_dir: str | os.PathLike = root / 'data/predictions' / "SegmentationCNN"
    results_dir: str | os.PathLike = root / 'data/predictions' / "UNetxx"
    selected_bands: None = None
    tile_size_gt: int = 4
    batch_size: int = 8
    seed: int = 12378921
    num_workers: int = 11
    model_path: str | os.PathLike = root / "models" / "UNetxx" / "last-run5.ckpt"



def main(options):
    """
    Prepares datamodule and loads model, then runs the evaluation loop

    Inputs:
        options: EvalConfig
            options for the experiment
    """
    # Load datamodule
    data_module = ESDDataModule(
        processed_dir=options.processed_dir,
        raw_dir=options.raw_dir,
        selected_bands=options.selected_bands,
    )

    # load model from checkpoint at options.model_path
    model = ESDSegmentation.load_from_checkpoint(checkpoint_path=str(options.model_path))

    # set the model to evaluation mode (model.eval())
    model.eval()

    # this is important because if you don't do this, some layers
    # will not evaluate properly

    # instantiate pytorch lightning trainer
    trainer = pl.Trainer()

    # run the validation loop with trainer.validate
    trainer.validate(model=model, datamodule=data_module)
    
    # run restitch_and_plot
    # restitch_and_plot(options, data_module, model, "Tile27", image_dir=options.results_dir)

    # for every subtile in options.processed_dir/Val/subtiles
    # run restitch_eval on that tile followed by picking the best scoring class
    # save the file as a tiff using tifffile
    # save the file as a png using matplotlib
    subtiles_dir = Path(options.processed_dir) / "Val" / "subtiles"
    subtile_files = list(subtiles_dir.glob("*.npz"))
    tiles = {path.stem.split('_')[0] for path in subtile_files}
    tiles = list(tiles)

    for parent_tile_id in tiles:
        _, _, stitched_prediction_subtile = restitch_eval(dir=subtiles_dir, tile_id=parent_tile_id,satellite_type="sentinel2", range_x=(0,4), range_y=(0,4), datamodule=data_module, model=model)
        # best scoring class
        best_class_prediction = np.argmax(stitched_prediction_subtile, axis=0)

        # freebie: plots the predicted image as a png with the correct colors
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list("Settlements", np.array(['#ff0000', '#0000ff', '#ffff00', '#b266ff']), N=4)
        fig, ax = plt.subplots(nrows=1, ncols=1)
        ax.imshow(best_class_prediction, vmin=-0.5, vmax=3.5,cmap=cmap)
        Path(options.results_dir / parent_tile_id).mkdir(parents=True, exist_ok=True)
        plt.savefig(options.results_dir / f"{parent_tile_id}" / f"{parent_tile_id}.png")

        # Save file as TIFF
        tifffile.imwrite(options.results_dir / f"{parent_tile_id}" / f"{parent_tile_id}.tiff", stitched_prediction_subtile)

        restitch_and_plot(
            options=options,
            datamodule=data_module,
            model=model,
            parent_tile_id=parent_tile_id,
            image_dir=options.results_dir / f"{parent_tile_id}"
        )

if __name__ == '__main__':
    config = EvalConfig()
    parser = ArgumentParser()

    parser.add_argument("--model_path", type=str, help="Model path.", default=config.model_path)
    parser.add_argument("--raw_dir", type=str, default=config.raw_dir, help='Path to raw directory')
    parser.add_argument("-p", "--processed_dir", type=str, default=config.processed_dir,
                        help=".")
    parser.add_argument("--results_dir", type=str, default=config.results_dir, help="Results dir")
    main(EvalConfig(**parser.parse_args().__dict__))