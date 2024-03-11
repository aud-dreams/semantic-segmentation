""" Augmentations Implemented as Callable Classes."""

import cv2
import numpy as np
import torch
import random
from typing import Dict


def apply_per_band(img, transform):
    """
    Helpful function to allow you to more easily implement
    transformations that are applied to each band separately.
    Not necessary to use, but can be helpful.
    """
    result = np.zeros_like(img)
    for band in range(img.shape[0]):
        transformed_band = transform(img[band].copy())
        result[band] = transformed_band

    return result


class Blur(object):
    """
    Blurs each band separately using cv2.blur

    Parameters:
        kernel: Size of the blurring kernel
        in both x and y dimensions, used
        as the input of cv.blur

    This operation is only done to the X input array.
    """

    def __init__(self, kernel=3):
        self.kernel = (kernel, kernel)

    def __call__(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Performs the blur transformation.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        # sample must have X and y in a dictionary format
        assert len(sample.keys()) == 2 and set(sample.keys()) == {"X", "y"}
        #  dimensions of img: (t, bands, tile_height, tile_width)
        x_blur = apply_per_band(sample["X"], lambda bands: cv2.blur(bands, self.kernel))
        return {"X": x_blur, "y": sample["y"]}


class AddNoise(object):
    """
    Adds random gaussian noise using np.random.normal.

    Parameters:
        mean: float
            Mean of the gaussian noise
        std_lim: float
            Maximum value of the standard deviation
    """

    def __init__(self, mean=0, std_lim=0.0):
        self.mean = mean
        self.std_lim = std_lim

    def __call__(self, sample):
        """
        Performs the add noise transformation.
        A random standard deviation is first calculated using
        random.uniform to be between 0 and self.std_lim

        Random noise is then added to each pixel with
        mean self.mean and the standard deviation
        that was just calculated

        The resulting value is then clipped using
        numpy's clip function to be values between
        0 and 1.

        This operation is only done to the X array.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        random_sd = random.uniform(0, self.std_lim)  # Random standard deviation
        noise = np.random.normal(self.mean, random_sd, sample["X"].shape)
        x_noise = np.clip(sample["X"] + noise, 0, 1)
        return {"X": x_noise, "y": sample["y"]}


class RandomVFlip(object):
    """
    Randomly flips all bands vertically in an image with probability p.

    Parameters:
        p: probability of flipping image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Performs the random flip transformation using cv.flip.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        x_flip = sample["X"]
        y_flip = sample["y"]
        if self.p >= random.random():
            x_flip = apply_per_band(x_flip, lambda bands: cv2.flip(bands, 0))
            y_flip = apply_per_band(y_flip, lambda bands: cv2.flip(bands, 0))
        return {"X": x_flip, "y": y_flip}


class RandomHFlip(object):
    """
    Randomly flips all bands horizontally in an image with probability p.

    Parameters:
        p: probability of flipping image.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        """
        Performs the random flip transformation using cv.flip.

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        x_flip = sample["X"]
        y_flip = sample["y"]
        if self.p >= random.random():
            x_flip = apply_per_band(x_flip, lambda bands: cv2.flip(bands, 1))
            y_flip = apply_per_band(y_flip, lambda bands: cv2.flip(bands, 1))
        return {"X": x_flip, "y": y_flip}


class ToTensor(object):
    """
    Converts numpy.array to torch.tensor
    """

    def __call__(self, sample):
        """
        Transforms all numpy arrays to tensors

        Input:
            sample: Dict[str, np.ndarray]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)

        Output:
            transformed: Dict[str, torch.Tensor]
                Has two keys, 'X' and 'y'.
                Each of them has shape (bands, width, height)
        """
        dict_list = sample.items()
        ret_dict = {}
        for key, val in dict_list:
            ret_dict[key] = torch.from_numpy(val)
        return ret_dict
