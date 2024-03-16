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
    # fill in the code here
    # from discord
    if not Path(Path(image_dir) / "viirs_max_projection.png").exists():
        Path(Path(image_dir) / "viirs_max_projection.png").touch(exist_ok=True)
        
    # minimax the data
    data = minmax_scale(viirs_stack)
    # flatten for purposes of plotting
    data = viirs_stack.flatten()

    plt.hist(data, bins=n_bins, color='blue', alpha=0.5, label='VIIRS', log=True)
    plt.title('VIIRS Histogram')

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
    # fill in the code here

    num_bands = sentinel1_stack.shape[2] # changed from [1]
    fig, ax = plt.subplots(2, 1, figsize=(50, 50))
    ax = ax.flatten()  
    for i in range(num_bands):
        data = sentinel1_stack[:, :, i, :, :].flatten()
        ax[i].hist(data, bins=n_bins, color='blue', alpha=0.5, label=f'Sentinel-2 Band {i+1}', log=True)
        ax[i].set_title(f'Sentinel-2 Band {i+1} Histogram')
    plt.tight_layout() # to make it more evenly proportionate 


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
    
    # fill in the code here
    # supposed to be multiple histograms, one for each band
    """
    data = sentinel2_stack.flatten()
    plt.hist(data, bins=n_bins, color='blue', alpha=0.5, label='Sentinel-2')
    """

    # iterate through each band
    #print("DEBUG: shape of sentinel2 stack for histogram:" + str(sentinel2_stack.shape))
    num_bands = sentinel2_stack.shape[2]
    fig, ax = plt.subplots(2, 6, figsize=(50, 50))
    ax = ax.flatten()  
    for i in range(num_bands):
        data = sentinel2_stack[:, :, i, :, :].flatten()
        ax[i].hist(data, bins=n_bins, color='blue', alpha=0.5, label=f'Sentinel-2 Band {i+1}', log=True)
        ax[i].set_title(f'Sentinel-2 Band {i+1} Histogram')
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
    # fill in the code here
    data = landsat_stack.flatten()
    plt.hist(data, bins=n_bins, color='blue', alpha=0.5, label='Landsat')
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
    # fill in the code here
    plt.hist(ground_truth.flatten(), bins=20, color='blue', alpha=0.5, label='Ground Truth')
    plt.title('Ground Truth Histogram')

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
    # fill in the code here
    plt.imshow(viirs)
    plt.title(plot_title)
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
    # fill in the code here
    
    # as many subplots as there are dates 
    # use metadata list for info per date

    # either len metadata or number of elements in viirs_stack[0]
    fig, ax = plt.subplots(len(metadata), 1, figsize=(50, 50))

    for i in range(len(metadata)):
        slice_data = np.squeeze(viirs_stack[i, :, :, :])
        ax[i].imshow(slice_data)
        ax[i].set_title(metadata[i].time)
    
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

    # in other words, this function is a wrapper for the preprocess functions

    preprocess_functions = {
        "sentinel2": preprocess_sentinel2,
        "sentinel1": preprocess_sentinel1,
        "landsat": preprocess_landsat,
        "viirs": preprocess_viirs
    }
    if satellite_type not in preprocess_functions:
        raise ValueError(f"Satellite type {satellite_type} is invalid.")
        
    preprocess_data = preprocess_functions[satellite_type](satellite_stack)
    return preprocess_data

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
    if len(bands_to_plot) > 3:
        raise ValueError("Cannot plot more than 3 bands.")

    # fill in the code here

    # to create an RGB composite is to retrieve the appropriate bands or channels

    # for VV-VH composite, after subtraction, minmax scale the image
    # Green is VV, Red is VH, Blue is VV-VH
    # plot VV, plot VH, plot VV-VH
    # cols are dates, rows are bands

    # NOTE: no need?
    # extract band_ids from metadata
    #band_ids = extract_band_ids(metadata) 

    # retrieve the information using band_ids?
    # in other words, create the image using stack

    # iterate thru each time, iterate thru each band 
    # one image per time, 3 channels per image
    
    num_plots = len(metadata)
    # init as many np arrays for images as there are dates
    list_images = [np.zeros((800, 800, 3)) for i in range(num_plots)]
    #print("DEBUG: processed_stack shape: " + str(processed_stack.shape))
    #print("Bands to plot: " + str(bands_to_plot))

    for i in range(num_plots):
        # image is 800 by 800 and 3 channel
        vv = processed_stack[i, 0, :, :]
        vh = processed_stack[i, 1, :, :]

        # quantile clip and minimax vv and vh
        vv = np.clip(vv, np.quantile(vv, 0.05), np.quantile(vv, 0.95))
        vh = np.clip(vh, np.quantile(vh, 0.05), np.quantile(vh, 0.95))

        # dummy dimension for minmax_scale

        vv = np.expand_dims(vv, axis=0)
        vh = np.expand_dims(vh, axis=0)
        vv = minmax_scale(vv, group_by_time=True)
        vh = minmax_scale(vh, group_by_time=True)
        # squeeze
        vv = np.squeeze(vv)
        vh = np.squeeze(vh)

        vv_vh = vv - vh

        # introduce dummy dimension for minmax_scale
        vv_vh = np.expand_dims(vv_vh, axis=0)
        vv_vh = minmax_scale(vv_vh, group_by_time=True)
        # squeeze 
        vv_vh = np.squeeze(vv_vh)

        # stack vv, vh, vv_vh
        list_images[i] = np.stack((vv, vh, vv_vh), axis=2)

    # make the subplots
    fig, ax = plt.subplots(num_plots, 1, figsize=(50, 50))
    for i in range(num_plots):
        ax[i].imshow(list_images[i])
        ax[i].set_title(metadata[i].time)

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

    for bands_list in bands_to_plot:
        for band in bands_list:
            if band not in band_mapping:
                raise ValueError(f"Band {band} is invalid.")


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
    # fill in the code here

    # band_mapping maps band to index of corresponding stack
    # bands to plot will tell you which images to plot!
    # Example: [['04', '03', '02'], ['08', '04', '03'], ['12', '8A', '04']]
    # First dimension is the date, second dimension is the bands for that date
    # num_plots equals number of dates and the number of bands
    # cols = dates, rows = bands
    
    num_plots = len(metadata) * len(bands_to_plot)
    fig, ax = plt.subplots(len(metadata), len(bands_to_plot), figsize=(50, 50))

    for i in range(len(metadata)):
        for j, bands in enumerate(bands_to_plot):
            image_list = []
            for k, band in enumerate(bands): 
                # if the band is in the list of bands to plot
                # retrieve the band from processed_stack
                band_index = band_mapping[band]
                band_itself = processed_stack[i, band_index, :, :]
                image_list.append(band_itself) 

            # stack the images in image_list
            image = np.stack(image_list, axis=2)
            ax[i, j].imshow(image)
            ax[i, j].set_title(f"Date: {metadata[i].time} - Bands: {bands}")


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
    # docstring should be updated so that it uses metadata?

    # bands can be found in metadata object

    band_identifiers = list(list())

    for file in metadata:
        band_identifiers.append(file.bands) # bands is a list of str
    return band_identifiers


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
    # fill in the code here
    
    data = np.squeeze(ground_truth)
    plt.imshow(data) 
    plt.title(plot_title)

    if image_dir is None:
        plt.show()
    else:
        plt.savefig(Path(image_dir) / "ground_truth.png")
        plt.close()
