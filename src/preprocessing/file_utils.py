"""
This module contains functions for loading satellite data from a directory of
tiles.
"""
from pathlib import Path 
from typing import Tuple, List, Set
import os
from itertools import groupby # great for grouping things, think hierarchical model of data
import re
from dataclasses import dataclass
import tifffile
import numpy as np


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

    # Relevant information is after the "VNP46A1_A" and before the ".tif"
    # from discord: it is implied that band is always 0.
    pattern = re.compile(r'DNB_VNP46A1_A(\d+).tif')
    
    band = "0"
    date = pattern.search(filename).group(1)
    return (date, band)


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
    
    # Relevant information is after the "GRDH_" and before the ".tif"
    # Date is the first 8 digits, band is the last 2 characters.
    filename = str(filename) # fixes some weird WindowsPath bug
    pattern = re.compile(r'S1A_IW_GRDH_(\d+)_([A-Z]{2}).tif')
   
    date = pattern.search(filename).group(1)
    band = pattern.search(filename).group(2)
    return (date, band)


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

    # Relevant information is after the "L2A_" and before the ".tif"
    # Date is the first 8 digits, band is the last 3 characters.
    # the 'B' is not included in the band.

    # NOTE: band can include both digits and letters!!!
    #print("DEBUG FOR EDA: FILENAME IS", filename)
    pattern = re.compile(r'L2A_(\d+)_B(\w{2}).tif')
    date = pattern.search(filename).group(1)
    band = pattern.search(filename).group(2)
    return (date, band)

def process_landsat_filename(filename: str) -> Tuple[str, str]:
    """
    This function takes in the filename of a Landsat file and outputs
    a tuple containing two strings, in the format (date, band)
    Example input: LC08_L1TP_2020-08-30_B9.tif
    Example output: ("2020-08-30", "B9")
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

    # Relevant information is after the "L1TP_" and before the ".tif"
    # Date is the first 10 digits, band is the last 2 characters.
    # Date includes the dashes.
    # Band can be 1 or 2 digits.
    pattern = re.compile(r'LC08_L1TP_(\d+-\d+-\d+)_B(\d{1,2}).tif')

    date = pattern.search(filename).group(1)
    band = pattern.search(filename).group(2)
    return (date, band)


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

    # also known as gt file
    return ("0", "0")


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
    # TODO: determine what exactly is a "Path" object for each satellite file

    path_list = []
    #pattern = re.compile(get_filename_pattern(satellite_type)) # the black box

    # iterator that returns all the files in the directory
    # doesn't use RE; instead, uses glob's unix style pattern

    tile_dir = Path(tile_dir) # cause tile_dir isn't actually being passed in as a Path obj but a str...
    pattern = get_filename_pattern(satellite_type)
    return [file for file in tile_dir.glob(pattern)]

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
    
    # What this does is return a string that can be used to match files using the standarized file names.
    patterns = {
        "viirs": 'DNB_VNP46A1_*',
        "sentinel1": 'S1A_IW_GRDH_*',
        "sentinel2": 'L2A_*',
        "landsat": 'LC08_L1TP_*',
        "gt": "groundTruth.tif"
    }

    try:
        return patterns[satellite_type]
    except: 
        KeyError(f"Invalid satellite type: {satellite_type}") # if input is not one of the above


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
    # a list of numpy arrays
    np_list = []
    for file in sat_files:
        file = tifffile.imread(file)
        np_list.append(file)
    return np_list


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
    grouping_func = get_grouping_function(satellite_type) # returns a function

    # Apply the grouping function to each file name to get the date and band
    date_and_band = [grouping_func(file) for file in file_names] # returns a list of tuples 

    # Sort the satellite data and file names based on the date and band
    # ie, group and sort
    grouped_data = sorted(zip(date_and_band, sat_data, file_names), key=lambda x: (x[0][0], x[0][1])) # lambda means "retrieve date and band from 1st tuple"

    # Initialize lists to store the stacked data and metadata
    stacked_sat_data = []
    metadata = []

    # Group the data by date
    # note to self: the date comes from index order
    for date, data in groupby(grouped_data, key=lambda x: x[0][0]):

        # Sort the group by band
        data = list(data) # to enable list comprehension

        # Extract the date and band, satellite data, and file names from the
        # sorted group
        bands, grouped_data, filenames = zip(*[(item[0][1], item[1], item[2]) for item in data])

        # Stack the satellite data along a new axis and append it to the list
        stacked_sat_data.append(np.stack(grouped_data, axis=0))
        # Create a Metadata object and append it to the list
        metadata.append(Metadata(satellite_type, list(filenames), 'Tile', list(bands), date))

    # Stack the list of satellite data arrays along a new axis to create a
    # 4D array with dimensions (time, band, height, width)
    stacked_sat_data = np.stack(stacked_sat_data, axis=0)

    # Return the stacked satelliet data and the list Metadata objects.
    return (stacked_sat_data, metadata)


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

    # ie if satellite_type is viirs, return process_viirs_filename
    # processing filenames will retrieve their date and band for our use
    functions = {
        "viirs": process_viirs_filename,
        "sentinel1": process_s1_filename,
        "sentinel2": process_s2_filename,
        "landsat": process_landsat_filename,
        "gt": process_ground_truth_filename
    }
    return functions[satellite_type]

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

    # can use set comprehension to get unique dates and bands
    return (set([date for date, band in metadata_keys]), set([band for date, band in metadata_keys]))

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
    
    # note to self: uses stack_satellite_data

    # open the directory and get the relevant satellite files
    path_list = get_satellite_files(tile_dir, satellite_type)

    # turn files into numpy arrays
    sat_data = read_satellite_files(path_list)

    # get the file names
    names = [file.name for file in path_list]

    # stack it
    return stack_satellite_data(sat_data, names, satellite_type) # returns the stacked data and metadata

def load_satellite_dir(
        data_dir: str | os.PathLike,
        satellite_type: str
        ) -> Tuple[np.ndarray, List[List[Metadata]]]:
    """
    Load all bands for a given satellite type from a directory of multiple tile files.
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

        # note to self: used for last step of stack_satellite_data
        # makes use of load_satellite by iterating through each tile directory
    
    multi_data = []
    multi_metadata = []

    # for each directory (use iterdir)
    data_dir = Path(data_dir)
    for tile_dir in data_dir.iterdir(): 
        if not tile_dir.is_dir():
            continue
        # load the satellite data
        data, metadata = load_satellite(tile_dir, satellite_type)
        # append to the list
        multi_data.append(data)
        multi_metadata.append(metadata)
    
    # stack all the data
    multi_data = np.stack(multi_data, axis=0)
    return (multi_data, multi_metadata)
