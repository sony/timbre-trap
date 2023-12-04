import numpy as np
import warnings
import scipy
import torch


__all__ = [
    'sum_gradient_norms',
    'average_gradient_norms',
    'get_max_gradient',
    'get_max_gradient_norm',
    'log_gradient_norms',
    'DataParallel',
    'filter_non_peaks',
    'threshold',
    'debug_nans'
]


def sum_gradient_norms(module):
    """
    Compute the cumulative gradient norm across all layers of a network.

    Parameters
    ----------
    module : torch.nn.Module
      Network containing gradients to track

    Returns
    ----------
    cumulative_norm : float
      Gradient norms summed across all layers
    """

    # Initialize cumulative norm
    cumulative_norm = 0.

    for layer, values in module.named_parameters():
        if values.grad is not None:
            # Compute the L2 norm of the gradients
            grad_norm = values.grad.norm(2).item()
            # Accumulate the norm of the gradients
            cumulative_norm += grad_norm

    return cumulative_norm


def average_gradient_norms(module):
    """
    Compute the average gradient norm across all layers of a network.

    Parameters
    ----------
    module : torch.nn.Module
      Network containing gradients to track

    Returns
    ----------
    average_norm : float
      Average gradient norm across all layers
    """

    # Initialize cumulative norm and layer count
    cumulative_norm, n_layers = 0., 0

    for layer, values in module.named_parameters():
        if values.grad is not None:
            # Compute the L2 norm of the gradients
            grad_norm = values.grad.norm(2).item()
            # Accumulate the norm of the gradients
            cumulative_norm += grad_norm
            # Increment layer count
            n_layers += 1

    # Compute average gradient norm
    average_norm = cumulative_norm / n_layers

    return average_norm


def get_max_gradient(module):
    """
    Determine the maximum gradient (magnitude) over all layers of a network.

    Parameters
    ----------
    module : torch.nn.Module
      Network containing gradients to track

    Returns
    ----------
    maximum_grad : float
      Maximum gradient (magnitude) over all layers
    """

    # Initialize maximum gradient
    maximum_grad = 0.

    for layer, values in module.named_parameters():
        if values.grad is not None:
            # Update tracked maximum gradient magnitude if necessary
            maximum_grad = max(maximum_grad, values.grad.abs().max().item())

    return maximum_grad


def get_max_gradient_norm(module):
    """
    Determine the maximum gradient norm over all layers of a network.

    Parameters
    ----------
    module : torch.nn.Module
      Network containing gradients to track

    Returns
    ----------
    maximum_norm : float
      Maximum gradient norm over all layers
    """

    # Initialize maximum norm
    maximum_norm = 0.

    for layer, values in module.named_parameters():
        if values.grad is not None:
            # Compute the L2 norm of the gradients
            grad_norm = values.grad.norm(2).item()
            # Update tracked maximum gradient norm
            maximum_norm = max(maximum_norm, grad_norm)

    return maximum_norm


def log_gradient_norms(module, writer, i=0, prefix='gradients/norm'):
    """
    Track gradient norms across each layer of a network.

    Parameters
    ----------
    module : torch.nn.Module
      Network containing gradients to track
    writer : SummaryWriter
      Results logger for tensorboard
    i : int
      Current iteration for logging
    prefix : str
      Tag prefix for logging
    """

    for layer, values in module.named_parameters():
        if values.grad is not None:
            # Compute the L2 norm of the gradients
            grad_norm = values.grad.norm(2).item()
            # Log the norm of the gradients for this layer
            writer.add_scalar(f'{prefix}/{layer}', grad_norm, i)


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
    extra_row = np.zeros(tuple(_arr.shape[:-2]) + (1, _arr.shape[-1]))

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
