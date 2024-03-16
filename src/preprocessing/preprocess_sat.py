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

    # multiple channels
    if img.ndim >= 3:
        # Apply the Gaussian filter to each channel
        return np.stack([gaussian_filter(img[:, :, channel], sigma) for channel in range(img.shape[2])], axis=-1)
    else:
        # single channel so just apply it directly
        return gaussian_filter(img, sigma)

    


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
        # by specifying axis=0, we are taking the quantile along the time dimension
        lower_bound = np.quantile(img_stack, clip_quantile, axis=0, keepdims=True)
        upper_bound = np.quantile(img_stack, 1 - clip_quantile, axis=0, keepdims=True)
    else:
        lower_bound = np.quantile(img_stack, clip_quantile)
        upper_bound = np.quantile(img_stack, 1 - clip_quantile)

    # determine the largest value below upper bound and smallest value above lower bound
    max_value = np.max(img_stack[img_stack < upper_bound])
    min_value = np.min(img_stack[img_stack > lower_bound])

    # clip
    processed_img = np.clip(img_stack, min_value, max_value)
    return processed_img    


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
        min_val = img.min(axis=(1,2), keepdims=True)
        max_val = img.max(axis=(1,2), keepdims=True)
    else:
        min_val = img.min(keepdims=True)
        max_val = img.max(keepdims=True)
    
    range_val = max_val - min_val 
    if np.all(range_val == 0):
        processed_img = np.zeros_like(img) # avoid dividing by 0! 
    else:
        processed_img = (img - min_val)/(range_val)

    # clip as needed
    scaled_img = np.clip(processed_img, 0, 1)

    return scaled_img

def brighten(img, alpha=0.13, beta=0):
    """
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
    processed_img = alpha*img[:, :, :, :] + beta
    processed_img[processed_img < 0] = 0
    processed_img[processed_img > 1] = 1
    return processed_img


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
    processed_band = np.power(band, 1/gamma)
    return processed_band


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
    # also, must clip (do it for each image individually)

    # for each time, clip every image (band) by input quantile 
    lower_bound = np.quantile(viirs_stack, clip_quantile, axis=(1, 2, 3), keepdims=True)
    upper_bound = np.quantile(viirs_stack, 1 - clip_quantile, axis=(1, 2, 3), keepdims=True)
    processed_stack = np.clip(viirs_stack, lower_bound, upper_bound) 

    # use np.max along time dimension (axis = 0)
    maxProj = np.max(processed_stack, axis=0)

    # minimax scale
    maxProj = minmax_scale(maxProj, group_by_time=True)
    return maxProj
    
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
        
    # Convert to dB scale
    processed_stack = 10 * np.log10(sentinel1_stack)
    
    # Clip outliers
    clipped_stack = quantile_clip(processed_stack, clip_quantile, group_by_time=False) # originally set to True

    # Apply a gaussian filter
    gauss_stack = per_band_gaussian_filter(clipped_stack, sigma=sigma)

    # Minmax scale
    mimimax_stack = minmax_scale(processed_stack, group_by_time=False) # originally set to True

    return mimimax_stack
    


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
    # clip outliers
    processed_stack = quantile_clip(sentinel2_stack, clip_quantile, group_by_time=False)
    # apply gamma correction
    processed_stack = gammacorr(processed_stack, gamma=gamma)
    # minmax scale
    processed_stack = minmax_scale(processed_stack, group_by_time=False)
    return processed_stack


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
    # clip outliers
    processed_stack = quantile_clip(landsat_stack, clip_quantile, group_by_time=True)
    # apply gamma correction
    processed_stack = gammacorr(processed_stack, gamma=gamma)
    # minmax scale
    processed_stack = minmax_scale(processed_stack, group_by_time=True)
    return processed_stack 


def preprocess_viirs(viirs_stack, clip_quantile=0.05) -> np.ndarray:
    """
    In this function we will preprocess viirs. The steps for preprocessing are
    the following:
        - Clip higher and lower quantile outliers per band per timestep
        - Minmax scale
    """
    # clip outliers
    processed_stack = quantile_clip(viirs_stack, clip_quantile, group_by_time=True)
    # minmax scale
    processed_stack = minmax_scale(processed_stack, group_by_time=True)
    return processed_stack
