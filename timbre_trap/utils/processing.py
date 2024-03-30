import numpy as np
import warnings
import scipy
import torch


__all__ = [
    'to_array',
    'debug_nans',
    'filter_non_peaks',
    'threshold'
]


def to_array(tensor):
    """
    Convert a PyTorch Tensor to a Numpy ndarray.

    Parameters
    ----------
    tensor : Tensor
      Arbitrary tensor data

    Returns
    ----------
    arr : ndarray
      Same data as Numpy ndarray
    """

    # Move to CPU, detach gradient, and convert to ndarray
    arr = tensor.cpu().detach().numpy()

    return arr


def debug_nans(tensor, tag='tensor'):
    """
    Check if a tensor contains NaNs and throw a warning when it happens.

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

    # Create a row of zeros
    zeros = np.zeros(tuple(_arr.shape[:-2]) + (1, _arr.shape[-1]))

    # Pad given array with extra rows to consider edge peaks
    padded_arr = np.concatenate((zeros, _arr, zeros), axis=-2)

    # Initialize array to hold filtered data
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
