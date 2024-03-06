"""
This module contains functions for loading satellite data from a directory of
tiles.
"""
from pathlib import Path
from typing import Tuple, List, Set
import os
from itertools import groupby
import re
from dataclasses import dataclass
import tifffile
import numpy as np

PATTERNS = {
            "viirs": 'DNB_VNP46A1_*',
            "sentinel1": 'S1A_IW_GRDH_*',
            "sentinel2": 'L2A_*',
            "landsat": 'LC08_L1TP_*',
            "gt": "groundTruth.tif"
        }

@dataclass
class Metadata:
    """
    A class to store metadata about a stack of satellite files from the same date.

    The attributes are the following:
    satellite_type: one of "viirs", "sentinel1", "sentinel2", "landsat", or "gt"
    file_name: a list of the original filenames of the satellite's bands
    tile_id: name of the tile directory, i.e., "Tile1", "Tile2", etc
    bands: a list of the names of the bands with correspondence to the
    indexes of the stack object, i.e. ["VH", "VV"] for sentinel-1
    time: time of the observations
    """
    satellite_type: str
    file_name: List[str]
    tile_id: str
    bands: List[str]
    time: str


def process_viirs_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a VIIRS file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: DNB_VNP46A1_A2020221.tif
    Example output: ("2020221", "0")

    Parameters
    ----------
    filename : str
        The filename of the VIIRS file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    match = re.search("DNB_VNP46A1_A(\d+)\.tif", filename)
    return (match.groups()[0], "0")

def process_s1_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-1 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: S1A_IW_GRDH_20200804_VV.tif
    Example output: ("20200804", "VV")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-1 file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    match = re.search(r"S1A_IW_GRDH_(\d+)_([A-Z]+)\.tif", filename)
    return match.groups()


def process_s2_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Sentinel-2 file and outputs
    a tuple containin two strings, in the format (date, band)

    Example input: L2A_20200816_B01.tif
    Example output: ("20200804", "01")

    Parameters
    ----------
    filename : str
        The filename of the Sentinel-2 file.

    Returns
    -------
    Tuple[str, str]
    """
    match = re.search(r"L2A_(\d+)_B(.{2})\.tif", filename)
    return match.groups()

def process_landsat_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band)

    Example input: LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "9")

    Parameters
    ----------
    filename : str
        The filename of the Landsat file.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    match = re.search(r"LC08_L1TP_(\d{4}-\d{2}-\d{2})_B(\d+)\.tif", filename)
    return match.groups()


def process_ground_truth_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of the ground truth file and returns
    ("0", "0"), as there is only one ground truth file.

    Example input: groundTruth.tif
    Example output: ("0", "0")

    Parameters
    ----------
    filename: str
        The filename of the ground truth file though we will ignore it.

    Returns
    -------
    Tuple[str, str]
        A tuple containing the date and band.
    """
    return (0, 0)


def get_satellite_files(tile_dir: Path, satellite_type: str) -> List[Path]:
    """
    Retrieve all satellite files matching the satellite type pattern.

    Parameters
    ----------
    tile_dir : Path
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    List[Path]
        A list of Path objects for each satellite file.
    """
    ret = []
    pattern = get_filename_pattern(satellite_type)
    tile_dir = Path(tile_dir)
    if tile_dir.is_dir():
        for file in tile_dir.glob(pattern):
            ret.append(file)
    else:
        raise FileNotFoundError
    return ret
    


def get_filename_pattern(satellite_type: str) -> str:
    """
    Return the filename pattern for the given satellite type.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    str
        The filename pattern for the given satellite type.
    """
    return PATTERNS[satellite_type]




def read_satellite_files(sat_files: List[Path]) -> List[np.ndarray]:
    """
    Read satellite files into a list of numpy arrays.

    Parameters
    ----------
    sat_files : List[Path]
        A list of Path objects for each satellite file.

    Returns
    -------
    List[np.ndarray]

    """
    ret = []
    for path in sat_files:
        img = tifffile.imread(path)
        ret.append(img) 
    return ret
    


def stack_satellite_data(
        sat_data: List[np.ndarray],
        file_names: List[str],
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Stack satellite data into a single array and collect filenames.

    Parameters
    ----------
    sat_data : List[np.ndarray]
        A list containing the image data for all bands with respect to
        a single satellite (sentinel-1, sentinel-2, landsat-8, or viirs)
        at a specific timestamp.
    file_names : List[str]
        A list of filenames corresponding to the satellite and timestamp.

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with dimensions
        (time, band, height, width) and a list of the filenames.
    """
    # Get the function to group the data based on the satellite type
    file_names = [str(x) for x in file_names]
    groupby_func = get_grouping_function(satellite_type)
    # print(len(sat_data), len(file_names))
    paired_list = []
    for i in range(len(sat_data)):
        paired_list.append((sat_data[i], file_names[i]))
    # print(groupby_func(paired_list[0][1]))
    paired_sorted = sorted(paired_list, key=lambda x: groupby_func(x[1]))
    daily_sat_data = []
    metadata_list = []
    for date, group in groupby(paired_sorted, lambda pair: groupby_func(pair[1])[0]):
        group = list(group)
        data_today = [x[0] for x in group]
        file_names_today = [x[1] for x in group]
        path = Path(file_names_today[0])
        band = []
        for file_name in file_names_today:
            band.append(groupby_func(file_name)[1])
        tile_id = path.parts[-2]
        # print(tile_id, band)
        meta_today = Metadata(satellite_type, file_names_today, tile_id, band, date)
        daily_sat_data.append(np.stack(data_today))
        metadata_list.append(meta_today)
    data_volume = np.stack(daily_sat_data)
    # print(data_volume.shape)
    return (data_volume, metadata_list)
        
    # Apply the grouping function to each file name to get the date and band

    # Sort the satellite data and file names based on the date and band

    # Initialize lists to store the stacked data and metadata

    # Group the data by date
        # Sort the group by band

        # Extract the date and band, satellite data, and file names from the
        # sorted group

        # Stack the satellite data along a new axis and append it to the list

        # Create a Metadata object and append it to the list

    # Stack the list of satellite data arrays along a new axis to create a
    # 4D array with dimensions (time, band, height, width)

    # Return the stacked satelliet data and the list Metadata objects.
    


def get_grouping_function(satellite_type: str):
    """
    Return the function to group satellite files by date and band.

    Parameters
    ----------
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    function
        The function to group satellite files by date and band.
    """
    # function should takes in a satellite file name and extract its date and 
    mapping = {
        "viirs": process_viirs_filename,
        "sentinel1": process_s1_filename,
        "sentinel2": process_s2_filename,
        "landsat": process_landsat_filename,
        "gt": process_ground_truth_filename
    }
    return mapping[satellite_type]

def get_unique_dates_and_bands(
        metadata_keys: Set[Tuple[str, str]]
        ) -> Tuple[Set[str], Set[str]]:
    """
    Extract unique dates and bands from satellite metadata keys.

    Parameters
    ----------
    metadata_keys : Set[Tuple[str, str]]
        A set of tuples containing the date and band for each satellite file.

    Returns
    -------
    Tuple[Set[str], Set[str]]
        A tuple containing the unique dates and bands.
    """
    unique_dates = set()
    unique_bands = set()
    for date, band in metadata_keys:
        unique_bands.add(band)
        unique_dates.add(date)
    return (unique_dates, unique_bands)



def load_satellite(
        tile_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[Metadata]]:
    """
    Load all bands for a given satellite type from a directory of tile files.

    Parameters
    ----------
    tile_dir : str or os.PathLike
        The Tile directory containing the satellite tiff files.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    Tuple[np.ndarray, List[Metadata]]
        A tuple containing the satellite data as a volume with
        dimensions (time, band, height, width) and a list of the filenames.
    """

    sat_files = get_satellite_files(tile_dir,satellite_type)
    sat_data = read_satellite_files(sat_files)
    return stack_satellite_data(sat_data, sat_files, satellite_type)


def load_satellite_dir(
        data_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[List[Metadata]]]:
    """
    Load all bands for a given satellite type fhttps://drive.google.com/file/d/FILE_ID/view?usp=sharing
rom a directory of multiple
    tile files.

    Parameters
    ----------
    data_dir : str or os.PathLike
        The directory containing the satellite tiles.
    satellite_type : str
        The type of satellite, one of "viirs", "sentinel1", "sentinel2",
        "landsat", "gt"

    Returns
    -------
    Tuple[np.ndarray, List[List[Metadata]]]
        A tuple containing the satellite data as a volume with
        dimensions (tile_dir, time, band, height, width) and a list of the
        Metadata objects.
    """
    data_per_tile = []
    tile_meta_data = []
    # data dir contains all tiles
    data_dir = Path(data_dir)
    for tile_dir in data_dir.iterdir():
        tile_data, meta_data = load_satellite(tile_dir, satellite_type)
        data_per_tile.append(tile_data)
        tile_meta_data.append(meta_data)
    return (np.stack(data_per_tile), tile_meta_data)
