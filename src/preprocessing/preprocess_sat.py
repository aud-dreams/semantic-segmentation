""" This code applies preprocessing functions on the IEEE GRSS ESD satellite data."""
import numpy as np
from scipy.ndimage import gaussian_filter


def per_band_gaussian_filter(img: np.ndarray, sigma: float = 1):
    """
    For each band in the image, apply a gaussian filter with the given sigma.
    The gaussian filter should be applied to each heightxwidth image individually.

    Parameters
    ----------
    img : np.ndarray
        The image to be filtered. The shape of the array is (time, band, height, width).
    sigma : float
        The sigma of the gaussian filter.

    Returns
    -------
    np.ndarray
        The filtered image. The shape of the array is (time, band, height, width).
    """
    ret_img = np.empty_like(img)
    
    for t in range(img.shape[0]):
        for b in range(img.shape[1]):
            ret_img[t, b]= gaussian_filter(img[t, b], sigma)
    return ret_img


def quantile_clip(img_stack: np.ndarray,
                  clip_quantile: float,
                  group_by_time=True
                  ) -> np.ndarray:
    """
    This function clips the outliers of the image stack by the given quantile.
    It calculates the top `clip_quantile` samples and the bottom `clip_quantile`
    samples, and sets any value above the top to the first value under the top value,
    and any value below the bottom to the first value above the top value.

    group_by_time affects how img_max and img_min are calculated, if
    group_by_time is true, the quantile limits are shared along the time dimension.
    Otherwise, the quantile limits are calculated individually for each image.

    Parameters
    ----------
    img_stack : np.ndarray
        The image stack to be clipped. The shape of the array is (time, band, height, width).
    clip_quantile : float
        The quantile to clip the outliers by. Value between 0 and 0.5.

    Returns
    -------
    np.ndarray
        The clipped image stack. The shape of the array is (time, band, height, width).
    """

    if group_by_time:
        lower_quantile = np.quantile(img_stack, clip_quantile, axis=(0, 2, 3), keepdims=True)
        upper_quantile = np.quantile(img_stack, 1 - clip_quantile, axis=(0, 2, 3), keepdims=True)
    else:
        lower_quantile = np.quantile(img_stack, clip_quantile, axis=(2, 3), keepdims=True)
        upper_quantile = np.quantile(img_stack, 1 - clip_quantile, axis=(2, 3), keepdims=True)

    img_stack_clipped = np.clip(img_stack, lower_quantile, upper_quantile)

    return img_stack_clipped


def minmax_scale(img: np.ndarray, group_by_time=True):
    """
    This function minmax scales the image stack to values between 0 and 1.
    This transforms any image to have a range between img_min to img_max
    to an image with the range 0 to 1, using the formula 
    (pixel_value - img_min)/(img_max - img_min).

    group_by_time affects how img_max and img_min are calculated, if
    group_by_time is true, the min and max are shared along the time dimension.
    Otherwise, the min and max are calculated individually for each image.
    
    Parameters
    ----------
    img : np.ndarray
        The image stack to be minmax scaled. The shape of the array is (time, band, height, width).
    group_by_time : bool
        Whether to group by time or not.

    Returns
    -------
    np.ndarray
        The minmax scaled image stack. The shape of the array is (time, band, height, width).
    """
    if group_by_time:
        img_min = np.min(img)
        img_max = np.max(img)
    else:
        img_min = np.min(img, axis=(1, 2, 3), keepdims=True)
        img_max = np.max(img, axis=(1, 2, 3), keepdims=True)

    # Apply the minmax scaling formula
    scaled_img = (img - img_min) / (img_max - img_min)

    return scaled_img


def brighten(img, alpha=0.13, beta=0):
    """
    Function to brighten the image.
    This is calculated using the formula new_pixel = alpha*pixel+beta.
    If a value of new_pixel falls outside of the [0,1) range,
    the values are clipped to be 0 if the value is under 0 and 1 if the value is over 1.

    Parameters
    ----------
    img : np.ndarray
        The image to be brightened. The shape of the array is (time, band, height, width).
        The input values are between 0 and 1.
    alpha : float
        The alpha parameter of the brightening.
    beta : float
        The beta parameter of the brightening.

    Returns
    -------
    np.ndarray
        The brightened image. The shape of the array is (time, band, height, width).
    """
    brightened_img = alpha * img + beta

    # Clip the values to ensure they are within [0, 1)
    brightened_img = np.clip(brightened_img, 0, 1)

    return brightened_img


def gammacorr(band, gamma=2):
    """
    This function applies a gamma correction to the image.
    This is done using the formula pixel^(1/gamma)

    Parameters
    ----------
    band : np.ndarray
        The image to be gamma corrected. The shape of the array is (time, band, height, width).
        The input values are between 0 and 1.
    gamma : float
        The gamma parameter of the gamma correction.

    Returns
    -------
    np.ndarray
        The gamma corrected image. The shape of the array is (time, band, height, width).
    """
    gamma_corrected = np.power(band, 1 / gamma)

    return gamma_corrected

def maxprojection_viirs(
        viirs_stack: np.ndarray,
        clip_quantile: float = 0.01
        ) -> np.ndarray:
    """
    This function takes a stack of VIIRS tiles and returns a single
    image that is the max projection of the tiles.

    The output value of the projection is such that 
    output[band,i,j] = max_time(input[time,band,i,j])
    i.e, the value of a pixel is the maximum value over all time steps.

    Parameters
    ----------
    tile_dir : str
        The directory containing the VIIRS tiles. The shape of the array is (time, band, height, width).

    Returns
    -------
    np.ndarray
        Max projection of the VIIRS stack, of shape (band, height, width)
    """
    # HINT: use the time dimension to perform the max projection over.
    # upper_bound = np.quantile(viirs_stack, 1-clip_quantile)
    # print(upper_bound)
    # lower_bound = np.quantile(viirs_stack, clip_quantile)
    lower_bound = np.quantile(viirs_stack, clip_quantile, axis=(1,2,3), keepdims=True)
    higher_bound = np.quantile(viirs_stack, 1 - clip_quantile, axis=(1,2,3), keepdims=True)
    clipped_stack = np.clip(viirs_stack, lower_bound, higher_bound)
    
    max_proj = np.max(clipped_stack, axis=0)
    max_proj = minmax_scale(max_proj, group_by_time=True)
    return max_proj


def preprocess_sentinel1(
        sentinel1_stack: np.ndarray,
        clip_quantile: float = 0.01,
        sigma=1
        ) -> np.ndarray:
    """
    In this function we will preprocess sentinel1. The steps for preprocessing
    are the following:
        - Convert data to dB (log scale)
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gaussian filter
        - Minmax scale
    """
    sentinel1_db = 10 * np.log10(sentinel1_stack)

    sentinel1_clipped = quantile_clip(sentinel1_db, clip_quantile, False)

    sentinel1_filtered = per_band_gaussian_filter(sentinel1_clipped, sigma)

    sentinel1_scaled = minmax_scale(sentinel1_filtered)

    return sentinel1_scaled


def preprocess_sentinel2(sentinel2_stack: np.ndarray,
                         clip_quantile: float = 0.05,
                         gamma: float = 2.2
                         ) -> np.ndarray:
    """
    In this function we will preprocess sentinel-2. The steps for
    preprocessing are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    clipped_stack = quantile_clip(sentinel2_stack, clip_quantile, group_by_time=False)

    gamma_corrected = gammacorr(clipped_stack, gamma)

    minmax_scaled = minmax_scale(gamma_corrected)

    return minmax_scaled


def preprocess_landsat(
        landsat_stack: np.ndarray,
        clip_quantile: float = 0.05,
        gamma: float = 2.2
        ) -> np.ndarray:
    """
    In this function we will preprocess landsat. The steps for preprocessing
    are the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Apply a gamma correction
        - Minmax scale
    """
    clipped_stack = quantile_clip(landsat_stack, clip_quantile, group_by_time=False)

    gamma_corrected = gammacorr(clipped_stack, gamma)

    minmax_scaled = minmax_scale(gamma_corrected)

    return minmax_scaled


def preprocess_viirs(viirs_stack, clip_quantile=0.05) -> np.ndarray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Minmax scale
    """
    clipped_stack = quantile_clip(viirs_stack, clip_quantile, group_by_time=False)

    minmax_scaled = minmax_scale(clipped_stack)

    return minmax_scaled
