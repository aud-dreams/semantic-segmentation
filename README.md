# HW01: Exploratory Data Analysis and Data Preprocessing
## TianYun Yuan

![Matplotlib](https://img.shields.io/badge/Matplotlib-%23ffffff.svg?style=for-the-badge&logo=Matplotlib&logoColor=black)

[![HW01 Tests](https://img.shields.io/badge/github%20actions-%232671E5.svg?style=for-the-badge&logo=githubactions&logoColor=white)](https://github.com/cs175cv-w2024/hw01-preprocessing-YS0meone/actions/workflows/hw01_tests.yml)

## Assignment Overview
In this homework we will familiarize ourselves with the different types of satellite data included in the IEEE GRSS ESD dataset. Additionaly we will:
- [ ] Setup your machine with the tools necessary using [venv](https://docs.python.org/3/library/venv.html) or python virtual environments.
- [ ] Write file utility functions for parsing filenames, retrieving satellite files, reading satellite data into NumPy arrays, and stacking satellite data for analysis. 
- [ ] Use of dataclasses to store metadata on each satellite stack comprised of multiple bands per timestamps in a tile directory from the `Train` set.
- [ ] Assess data quality by generating histograms for specific satellite data over multiple tile directories in the `Train` set to understand the distribution of pixel values across different spectral bands and times.
- [ ] Preprocessing the data based on histogram output specific to satellite type.
- [ ] Visualize of three channel band combinations for Sentinel-1, Sentinel-2, and Landsat-8 images.
- [ ] Explore VIIRS nighttime data over all time stamps per tile available using maximum projection representation.

## Setting Up Your Virtual Project Environment
To make sure you download all the packages to begin this homework assignment we will utilize a Python virtual environment which is an isolated environment that allows you to run the homework with its own dependencies and libraries independent of other Python projects you may be working on. Here's how to set it up:

1. Navigate to the hw01 project directory in your terminal.

2. Create a virtual environment:
   
   `python3 -m venv esdenv`
3. Activate the virtual environment:
   * On macOS and Linux:
  
        `source esdenv/bin/activate`
   * On Windows:
  
        `.\esdenv\Scripts\activate`
4. Install the required packages:
    `pip install -r requirements.txt`

To deactivate the virtual environment, type `deactivate`.

## The Data
### Description
** The following description is taken directly from the IEEE GRSS 2021 Challenge [website](https://www.grss-ieee.org/community/technical-committees/2021-ieee-grss-data-fusion-contest-track-dse/).

The IEEE GRSS 2021 ESD dataset is composed of 98 tiles of 800×800 pixels, distributed respectively across the training, validation and test sets as follows: 60, 19, and 19 tiles. Each tile includes 98 channels from the below listed satellite images. Please note that all the images have been resampled to a Ground Sampling Distance (GSD) of 10 m. Thus each tile corresponds to a 64km2 area.

### Satellite Data
#### Sentinel-1 polarimetric SAR dataset

2 channels corresponding to intensity values for VV and VH polarization at a 5×20 m spatial resolution scaled to a 10×10m spatial resolution.

File name prefix: “S1A_IW_GRDH_*.tif”
Size : 2.1 GB (float32)
Number of images : 4
Acquisition mode : Interferometric Wide Swath
Native resolution : 5x20m
User guide : [link](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-1-sar/acquisition-modes/interferometric-wide-swath)

#### Sentinel-2 multispectral dataset

12 channels of reflectance data covering VNIR and SWIR ranges at a GSD of 10 m, 20 m, and 60 m. The cirrus band 10 is omitted, as it does not contain ground information.

File name prefix: “L2A_*.tif”
Size : 6,2 GB (uint16)
Number of images : 4
Level of processing : 2A
Native Resolution : [link](https://sentinel.esa.int/web/sentinel/technical-guides/sentinel-2-msi/msi-instrument)
User guide : [link](https://sentinel.esa.int/web/sentinel/user-guides/sentinel-2-msi/processing-levels/level-2)


#### Landsat 8 multispectral dataset

11 channels of reflectance data covering VNIR, SWIR, and TIR ranges at a GSD of 30m and 100 m, and a Panchromatic band at a GSD of 15m.

File name prefix: “LC08_L1TP_*.tif”
Size : 8,5 GB (float32)
Number of images : 3
Sensor used : OLI and TIRS
Native Resolution : [link](https://landsat.gsfc.nasa.gov/landsat-8/landsat-8-overview/)
User guide : [link](https://www.usgs.gov/core-science-systems/nli/landsat/landsat-8-data-users-handbook)

#### The Suomi Visible Infrared Imaging Radiometer Suite (VIIRS) night time dataset

The Day-Night Band (DNB) sensor of the VIIRS (Visible Infrared Imaging Radiometer Suite) provides on 1 channel, the global daily measurements of nocturnal visible and near-infrared (NIR) light at a GSD of 750 m. The VNP46A1 product is a corrected version of the original DNB data, and is at a 500m GSD resolution.

File name prefix: “DNB_VNP46A1_*.tif”
Size : 1,2 GB(uint16)
Number of images : 9
Product Name : VNP46A1’s 500x500m sensor radiance dataset
Native resolution : 750m (raw resolution)
User sheet : [link](https://viirsland.gsfc.nasa.gov/PDF/VIIRS_BlackMarble_UserGuide.pdf)

### Semantic Labels
The provided training data is split across 60 folders named TileX, X being the tile number. Each folder includes 100 files. 98 files correspond to the satellite images listed earlier.

We also provide reference information (‘groundTruth.tif’ file) for each tile. Please note that the labeling has been performed as follows:

#### Human settlement: 
If a building is present in a patch of 500×500m, this area is considered to have human settlement

#### Presence of Electricity: 
If a fraction of a patch of 500×500m is illuminated, this area is considered to be illuminated regardless of the size of the illuminated area fraction.
The reference file (‘groundTruth.tif’) is 16×16 pixels large, with a resolution of 500m corresponding to the labelling strategy described above. The pixel values (1, 2, 3 and 4) correspond to the four following classes:

1: Human settlements without electricity (Region of Interest): Color ff0000
2: No human settlements without electricity: Color 0000ff
3: Human settlements with electricity: Color ffff00
4: No human settlements with electricity: Color b266ff
An additional reference file ( ‘groundTruthRGB.png’ ) is provided at 10m resolution in RGB for easier visualization.

# Directions :mega:
### Retrieving The Dataset

Please download and unzip the `dfc2021_dse_train.zip` saving the `Train` directory into the `data/raw` directory. You do not need to worry about registering to get the data from the IEEE DataPort as we have already downloaded it for you.
The zip file is available at the following [url](https://drive.google.com/file/d/1mVDV9NkmyfZbkSiD5lkskv_MwOuYxiog/view?usp=sharing).

## File Utility Functions
`src/preprocessing/file_utils.py` defines a module for loading and processing satellite imagery data. The code includes a `Metadata` class to store satellite file metadata and several functions to process filenames from different satellite types (e.g., VIIRS, Sentinel-1, Sentinel-2, Landsat), extract metadata, load satellite files, and stack satellite data. Here's a brief description of what each function you will implement does:

- `Metadata` class: Stores metadata about a satellite file, including satellite type, filename, tile ID, bands, and time.

- `process_viirs_filename`, `process_s1_filename`, `process_s2_filename`, `process_landsat_filename`: These functions take a filename string as input and extract the date and band information in a tuple format using pattern matching.

- `process_ground_truth_filename`: Returns a tuple with default values since there is only one ground truth file.

- `get_satellite_files`: Retrieves a list of satellite files from a given directory and satellite type.

- `get_filename_pattern`: Returns the filename pattern for a specific satellite type.

- `read_satellite_files`: Reads satellite files into a list of NumPy arrays.

- `stack_satellite_data`: Stacks satellite data into a single array and collects filenames, grouping them by date and band.

- `get_grouping_function`: Returns the appropriate function to group satellite files by date and band, based on the satellite type.

- `get_unique_dates_and_bands`: Extracts unique dates and bands from satellite metadata keys.

- `load_satellite`: Loads all bands for a given satellite type from a directory of tile files, returning a tuple with the satellite data as a NumPy array and a list of filenames.

- `load_satellite_dir`: Loads all bands for a given satellite type from a directory containing multiple tile files, returning a tuple with the satellite data as a NumPy array and a list of `Metadata` objects.

When completing the code, we encourage you to explore how each function is used and how the data is transformed at each step of the process.

## Preprocessing the Data
`src/preprocessing/preprocess_sat.py` is a module to preprocess satellite imagery by performing image enhancement and normalization techniques to prepare the data for exploratory data analysis and visualization using matplotlib. This step will be important for our machine learning model to ensure that our data attributes, in this case the spectral bands of each satellite image can provide context without one spectral band overpowering the others due to outliers. Here is a summary of the functions you will implement:

- `per_band_gaussian_filter`: Applies a Gaussian filter to each band within the image to smooth out noise.

- `quantile_clip`: Clips the outliers in the image stack by a specified quantile to reduce the effect of extreme values.

- `minmax_scale`: Scales the image data to a range of 0 to 1 on a per-band basis, where the minimum value of the band becomes 0 and the maximum value becomes 1.

- `brighten`: Adjusts the brightness of the image using an alpha (gain) and beta (bias) parameter.

- `gammacorr`: Applies gamma correction to adjust the luminance of the image.

- `maxprojection_viirs`: Takes a stack of VIIRS tiles and creates a single image representing the maximum value projection, which is useful for visualizing the brightest areas.

- `preprocess_sentinel1`: Preprocesses Sentinel-1 data by converting to a logarithmic scale (dB), clipping quantile outliers, applying a Gaussian filter, and scaling the data using min-max normalization.

- `preprocess_sentinel2`: Preprocesses Sentinel-2 data by clipping quantile outliers, applying gamma correction, and performing min-max scaling.

- `preprocess_landsat`: Preprocesses Landsat data similarly to Sentinel-2, with clipping of quantile outliers, gamma correction, and min-max scaling.

- `preprocess_viirs`: Preprocesses VIIRS data by clipping quantile outliers and performing min-max scaling.

## Visualizing the Data
`src/visualization/plot_utils.py` should include a collection of functions that can handle different satellite data types such as VIIRS, Sentinel-1, Sentinel-2, and Landsat, and perform operations like plotting histograms of the data, creating RGB composites, and plotting ground truth data. We would like for you to implement the following:

- `plot_viirs_histogram`: Plots a histogram of all VIIRS values.
- `plot_sentinel1_histogram`: Plots histograms of Sentinel-1 values, both in linear and logarithmic scales.
- `plot_sentinel2_histogram`: Plots a histogram of Sentinel-2 values with a logarithmic y-axis.
- `plot_landsat_histogram`: Plots a histogram of Landsat values over all tiles present in the stack.
- `plot_gt_counts`: Plots a bar chart of ground truth values.
- `plot_viirs`: Plots a single VIIRS image.
- `plot_viirs_by_date`: Plots VIIRS images by date in subplots.
- `preprocess_data`: Calls specific preprocessing functions based on the satellite type.
- `create_rgb_composite_s1`: Creates an RGB composite for Sentinel-1 images.
- `validate_band_identifiers`: Validates band identifiers for plotting.
- `plot_images`: Plots satellite images based on specified bands.
- `plot_satellite_by_bands`: Plots satellite images by bands in subplots.
- `extract_band_ids`: Extracts band identifiers from metadata.
- `plot_ground_truth`: Plots a ground truth image.

## Scripts to Run to See Visualizations
We encourage you to manipulate `scripts/eda.py` to run your functions and compare different preprocessing steps or visualizing different types of satellite data and tri-band combinations to understand their characteristics. You may also observe how changes in parameters like `n_bins` or `clip_quantile` affect the visual output. To run the script as a package run the following command:

`python -m scripts.eda`

## Files to Work On :white_check_mark:
- `src/preprocessing/file_utils.py`
- `src/preprocessing/preprocess_sat.py`
- `src/visualization/plot_utils.py`

## NOTE
- It is required that you add your name and github actions workflow badge to your readme.
- Check the logs from github actions to verify the correctness of your program.
- The initial code will not work. You will have to write the necessary code and fill in the gaps.
- Commit all changes as you develop the code in your individual private repo. Please provide descriptive commit messages and push from local to your repository. If you do not stage, commit, and push git classroom will not receive your code at all.
- Make sure your last push is before the deadline. Your last push will be considered as your final submission.
- There is no partial credit for code that does not run.
- If you need to be considered for partial grade for any reason (failing tests on github actions,etc). Then message the staff on discord before the deadline. Late requests may not be considered.

## References
[GH Badges](https://docs.github.com/en/actions/monitoring-and-troubleshooting-workflows/adding-a-workflow-status-badge)
[Markdown Badges](https://github.com/Ileriayo/markdown-badges)
