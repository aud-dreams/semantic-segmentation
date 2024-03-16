""" This module contains functions for plotting satellite images. """
import os
from pathlib import Path
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from ..preprocessing.file_utils import Metadata
from ..preprocessing.preprocess_sat import minmax_scale
from ..preprocessing.preprocess_sat import (
    preprocess_sentinel1,
    preprocess_sentinel2,
    preprocess_landsat,
    preprocess_viirs
)


def plot_viirs_histogram(
        viirs_stack: np.ndarray,
        image_dir: None | str | os.PathLike = None,
        n_bins=100
        ) -> None:
    """
    This function plots the histogram over all VIIRS values.
    note: viirs_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    processed_viirs = viirs_stack.flatten()
    fig, ax = plt.subplots()
    ax.hist(processed_viirs, bins=n_bins)
    ax.set_title("VIIRS Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    # fill in the code here
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "VIIRS_histogram.png")
        plt.close()


def plot_sentinel1_histogram(
        sentinel1_stack: np.ndarray,
        metadata: List[Metadata],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the Sentinel-1 histogram over all Sentinel-1 values.
    note: sentinel1_stack is a 4D array of shape (time, band, height, width)

    Parameters
    ----------
    sentinel1_stack : np.ndarray
        The Sentinel-1 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    processed_s1 = sentinel1_stack.flatten()
    fig, ax = plt.subplots()
    ax.hist(processed_s1, bins=n_bins)
    ax.set_title("S1 Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel1_histogram.png")
        plt.close()


def plot_sentinel2_histogram(
        sentinel2_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20) -> None:
    """
    This function plots the Sentinel-2 histogram over all Sentinel-2 values.

    Parameters
    ----------
    sentinel2_stack : np.ndarray
        The Sentinel-2 image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-2 image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    processed_s2 = sentinel2_stack.flatten()
    fig, ax = plt.subplots()
    ax.hist(processed_s2, bins=n_bins)
    ax.set_title("S2 Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    # fill in the code here
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "sentinel2_histogram.png")
        plt.close()


def plot_landsat_histogram(
        landsat_stack: np.ndarray,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None,
        n_bins=20
        ) -> None:
    """
    This function plots the landsat histogram over all landsat values over all
    tiles present in the landsat_stack.

    Parameters
    ----------
    landsat_stack : np.ndarray
        The landsat image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the landsat image stack.
    image_dir : None | str | os.PathLike
        The directory to save the image to.
    n_bins : int
        The number of bins to use in the histogram.

    Returns
    -------
    None
    """
    processed_la = landsat_stack.flatten()
    fig, ax = plt.subplots()
    ax.hist(processed_la, bins=n_bins)
    ax.set_title("S2 Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    ax.set_yscale("log")
    # fill in the code here
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "landsat_histogram.png")
        plt.close()


def plot_gt_counts(ground_truth: np.ndarray,
                   image_dir: None | str | os.PathLike = None
                   ) -> None:
    """
    This function plots the ground truth histogram over all ground truth
    values over all tiles present in the groundTruth_stack.

    Parameters
    ----------
    groundTruth : np.ndarray
        The ground truth image stack volume.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """

    processed_gt = ground_truth.flatten()
    fig, ax = plt.subplots()
    ax.hist(processed_gt)
    ax.set_title("GT Histogram")
    ax.set_xlabel("Value")
    ax.set_ylabel("Frequency")
    # fill in the code here
    # fill in the code here
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth_histogram.png")
        plt.close()

def plot_viirs(
        viirs: np.ndarray, plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """ This function plots the VIIRS image.

    Parameters
    ----------
    viirs : np.ndarray
        The VIIRS image.
    plot_title : str
        The title of the plot.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    fig, ax = plt.subplots()
    cax = ax.imshow(viirs)
    ax.set_title("VIIRS Image")
    fig.colorbar(cax)
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_max_projection.png")
        plt.close()

def plot_viirs_by_date(
        viirs_stack: np.array,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None) -> None:
    """
    This function plots the VIIRS image by band in subplots.

    Parameters
    ----------
    viirs_stack : np.ndarray
        The VIIRS image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the VIIRS image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # We should assume that the image is just one tile with multiple date and band one
    squeezed_stack = viirs_stack.squeeze(1)
    num_dates = squeezed_stack.shape[0]
    unit = 8
    fig, ax = plt.subplots(1, num_dates, figsize=(num_dates * unit, unit))
    for date_idx in range(num_dates):
        cax = ax[date_idx].imshow(squeezed_stack[date_idx,:,:])
        fig.colorbar(cax, ax=ax[date_idx], orientation="vertical")
        date = metadata[date_idx].time
        ax[date_idx].set_title(f"{date}", fontsize=40)
    plt.tight_layout()
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "viirs_plot_by_date.png")
        plt.close()


def preprocess_data(
        satellite_stack: np.ndarray,
        satellite_type: str
        ) -> np.ndarray:
    """
    This function preprocesses the satellite data based on the satellite type.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    np.ndarray
    """
    preprocess_func_mapping = {
        "sentinel2": preprocess_sentinel2,
        "sentinel1": preprocess_sentinel1,
        "landsat": preprocess_landsat,
        "viirs": preprocess_viirs
    }
    preprocess_func = preprocess_func_mapping[satellite_type]
    return preprocess_func(satellite_stack)

def minmax_helper(img: np.ndarray):
    img_min = np.min(img)
    img_max = np.max(img)
    scaled_img = (img - img_min) / (img_max - img_min)
    return scaled_img

def create_rgb_composite_s1(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function creates an RGB composite for Sentinel-1.
    This function needs to extract the band identifiers from the metadata
    and then create the RGB composite. For the VV-VH composite, after
    the subtraction, you need to minmax scale the image.

    Parameters
    ----------
    processed_stack : np.ndarray
        The Sentinel-1 image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot. Cannot accept more than 3 bands.
    metadata : List[List[Metadata]]
        The metadata for the Sentinel-1 image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    # assuming S1
    num_date = processed_stack.shape[0]
    num_band = processed_stack.shape[1]
    daily_composite = []
    unit = 8
    fig, ax = plt.subplots(1, num_date, figsize=(unit * num_date, unit))
    
    for date_idx in range(num_date):
        band_list = metadata[date_idx].bands
        date = metadata[date_idx].time
        band2arr = {}
        for band_idx in range(num_band):
            band = band_list[band_idx]
            if band == "VV":
                band2arr["VV"] = processed_stack[date_idx, band_idx, :, :]
            elif band == "VH":
                band2arr["VH"] = processed_stack[date_idx, band_idx, :, :]
        band2arr["VV-VH"] = minmax_helper(band2arr["VV"] - band2arr["VH"])
        # print(band2arr.keys())
        composite_stack = [band2arr[b] for b in bands_to_plot[0]]
        rgb_img = np.dstack(composite_stack)
        cax = ax[date_idx].imshow(rgb_img)
        ax[date_idx].set_title(date, fontsize=16)
    if len(bands_to_plot) > 3:
        raise ValueError("Cannot plot more than 3 bands.")

    # fill in the code here

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "plot_sentinel1.png")
        plt.close()


def validate_band_identifiers(
          bands_to_plot: List[List[str]],
          band_mapping: dict) -> None:
    """
    This function validates the band identifiers.

    Parameters
    ----------
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.

    Returns
    -------
    None
    """
    
    for band_list in bands_to_plot:
        for band in band_list:
            if band not in band_mapping.keys():
                raise ValueError("The band mapping does not contain the band to plot!")

def plot_images(
        processed_stack: np.ndarray,
        bands_to_plot: List[List[str]],
        band_mapping: dict,
        metadata: List[List[Metadata]],
        image_dir: None | str | os.PathLike = None
        ):
    """
    This function plots the satellite images.
    
    Parameters
    ----------
    processed_stack : np.ndarray
        The satellite image stack volume.
    bands_to_plot : List[List[str]]
        The bands to plot.
    band_mapping : dict
        The band mapping.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    image_dir : None | str | os.PathLike
        The directory where the image should be saved.

    Returns
    -------
    None
    """
    num_date = processed_stack.shape[0]
    num_band = processed_stack.shape[1]
    num_plot_each_date = len(bands_to_plot)
    unit = 8
    fig, ax = plt.subplots(num_date, num_plot_each_date, figsize=(unit * num_plot_each_date, unit * num_date))
    rgb_stack = []
    for date in range(num_date):
        for i in range(num_plot_each_date):
            rgb_stack = []
            for band in bands_to_plot[i]:
                rgb_stack.append(processed_stack[date, band_mapping[band], :, :])
            composite = np.dstack(rgb_stack)
            ax[date][i].imshow(composite)        

    plt.tight_layout()

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(
            Path(image_dir) / f"plot_{metadata[0].satellite_type}.png"
            )
        plt.close()


def plot_satellite_by_bands(
        satellite_stack: np.ndarray,
        metadata: List[Metadata],
        bands_to_plot: List[List[str]],
        satellite_type: str,
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the satellite image by band in subplots.

    Parameters
    ----------
    satellite_stack : np.ndarray
        The satellite image stack volume.
    metadata : List[List[Metadata]]
        The metadata for the satellite image stack.
    bands_to_plot : List[List[str]]
        The bands to plot.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    None
    """
    
    processed_stack = preprocess_data(satellite_stack, satellite_type)

    if satellite_type == "sentinel1":
        create_rgb_composite_s1(processed_stack, bands_to_plot, metadata, image_dir=image_dir)
    else:
        band_ids_per_timestamp = extract_band_ids(metadata)
        all_band_ids = [band_id for timestamp in band_ids_per_timestamp for
                        band_id in timestamp]
        unique_band_ids = sorted(list(set(all_band_ids)))
        band_mapping = {band_id: idx for
                        idx, band_id in enumerate(unique_band_ids)}
        validate_band_identifiers(bands_to_plot, band_mapping)
        plot_images(
            processed_stack,
            bands_to_plot,
            band_mapping,
            metadata,
            image_dir
            )


def extract_band_ids(metadata: List[Metadata]) -> List[List[str]]:
    """
    Extract the band identifiers from file names for each timestamp based on
    satellite type.

    Parameters
    ----------
    file_names : List[List[str]]
        A list of file names.
    satellite_type : str
        The satellite type. One of "sentinel2", "sentinel1",
        "landsat", or "viirs".

    Returns
    -------
    List[List[str]]
        A list of band identifiers.
    """
    band_ids_per_timestamp = [] 
    for date_idx in range(len(metadata)):
        band_lists = metadata[date_idx].bands
        band_ids_per_timestamp.append(band_lists)
    return band_ids_per_timestamp


def plot_ground_truth(
        ground_truth: np.array,
        plot_title: str = '',
        image_dir: None | str | os.PathLike = None
        ) -> None:
    """
    This function plots the groundTruth image.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles.
    """
    squeezed = ground_truth.squeeze()
    fig, ax = plt.subplots()
    cax = ax.imshow(squeezed)
    fig.colorbar(cax)
    ax.set_title("Ground Truth Image")
    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()
