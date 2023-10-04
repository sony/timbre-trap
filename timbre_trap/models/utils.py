import numpy as np
import warnings
import scipy
import torch


__all__ = [
    'DataParallel',
    'filter_non_peaks',
    'threshold',
    'debug_nans'
]


class DataParallel(torch.nn.DataParallel):
    """
    A custom nn.DataParallel module to retain method names and attributes.
    """

    def __getattr__(self, name):
        try:
            # Check DataParallel
            return super().__getattr__(name)
        except AttributeError:
            # Check the wrapped model
            return getattr(self.module, name)


def filter_non_peaks(_arr):
    """
    Remove any values that are not peaks along the vertical axis.

    Parameters
    ----------
    _arr : ndarray (... x H x W)
      Original data

    Returns
    ----------
    arr : ndarray (... x H x W)
      Data with non-peaks removed
    """

    # Create an extra row to all for edge peaks
    extra_row = np.zeros(tuple([1] * len(_arr.shape[:-1])) + (_arr.shape[-1],))

    # Pad the given array with extra rows
    padded_arr = np.concatenate((extra_row, _arr, extra_row), axis=-2)

    # Initialize an array to hold filtered data
    arr = np.zeros(padded_arr.shape)

    # Determine which indices correspond to peaks
    peaks = scipy.signal.argrelmax(padded_arr, axis=-2)

    # Transfer peaks to new array
    arr[peaks] = padded_arr[peaks]

    # Remove padded rows
    arr = arr[..., 1 : -1, :]

    return arr


def threshold(_arr, t=0.5):
    """
    Binarize data based on a given threshold.

    Parameters
    ----------
    _arr : ndarray
      Original data
    t : float [0, 1]
      Threshold value

    Returns
    ----------
    arr : ndarray
      Binarized data
    """

    # Initialize an array to hold binarized data
    arr = np.zeros(_arr.shape)
    # Set values above threshold to one
    arr[_arr >= t] = 1

    return arr


def debug_nans(tensor, tag='tensor'):
    """
    Check if a tensor contains NaNs and throw warnings when this happens.

    Parameters
    ----------
    tensor : Tensor
      Arbitrary tensor data
    tag : str
      Name of the tensor for warning message

    Returns
    ----------
    contains : bool
      Whether input tensor contains NaN values
    """

    # Default indicator
    contains = False

    if torch.sum(tensor.isnan()):
        # Throw a warning if the tensor contains NaNs
        warnings.warn(f'{tag} contains NaNs!!!')

        # Set indicator
        contains = True

    return contains
