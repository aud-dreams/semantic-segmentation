import pyprojroot
import sys

root = pyprojroot.here()
sys.path.append(str(root))
from src.esd_data.datamodule import ESDDataModule
from src.models.unsupervised.dim_reduction import (
    preprocess_for_dim_reduction,
    perform_PCA,
    perform_TSNE,
    perform_UMAP,
)
from src.visualization.plot_utils_hw02 import plot_2D_scatter_plot

import os
import sys
import numpy as np
import torch


def main():
    import pyprojroot

    root = pyprojroot.here()
    processed_dir = root / "data" / "processed" / "Train" / "subtiles"
    raw_dir = root / "data" / "raw" / "Train"
    esd_dm = ESDDataModule(
        processed_dir,
        raw_dir,
        batch_size=1,
        selected_bands={"sentinel1": ["VV", "VH"], "viirs_maxproj": ["0"]},
        tile_size_gt=1,
    )
    esd_dm.prepare_data()
    esd_dm.setup("fit")

    X_flat, y_flat = preprocess_for_dim_reduction(esd_datamodule=esd_dm)

    # uncomment the following lines to perform PCA, TSNE, and UMAP

    print("Performing PCA")
    X_pca, pca = perform_PCA(X_flat, 2)
    plot_2D_scatter_plot(X_pca, y_flat, "PCA", root / 'plots')

    print("Performing TSNE")
    X_tsne, tsne = perform_TSNE(X_flat, 2)
    plot_2D_scatter_plot(X_tsne, y_flat, "TSNE", root / "plots")

    print("Performing UMAP")
    X_umap, umap = perform_UMAP(X_flat, 2)
    plot_2D_scatter_plot(X_umap, y_flat, "UMAP", root / 'plots')


if __name__ == "__main__":
    main()
