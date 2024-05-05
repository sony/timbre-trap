from copy import deepcopy
from math import cos, pi

import numpy as np
import mir_eval
import random
import torch
import sys


__all__ = [
    'seed_everything',
    'print_and_log',
    'DataParallel',
    'CosineWarmup',
    'sum_gradient_norms',
    'average_gradient_norms',
    'get_max_gradient',
    'get_max_gradient_norm',
    'log_gradient_norms',
    'MultipitchEvaluator'
]


def seed_everything(seed):
    """
    Set all necessary seeds for PyTorch at once.

    WARNING: the number of workers in the training loader affects behavior:
             this is because each sample will inevitably end up being processed
             by a different worker if num_workers is changed, and each worker
             has its own random seed

    Parameters
    ----------
    seed : int
      Seed to use for random number generation
    """

    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)


def print_and_log(text, path=None):
    """
    Print a string to the console and optionally log it to a specified file.

    Parameters
    ----------
    text : str
      Text to print/log
    path : str or None (Optional)
      Path to file to write text
    """

    # Print the text to console
    print(text)

    if path is not None:
        with open(path, 'a') as f:
            # Append the text to the file
            print(text, file=f)


class DataParallel(torch.nn.DataParallel):
    """
    A custom nn.DataParallel module to retain method names and attributes.
    """

    def __getattr__(self, name):
        try:
            # Check DataParallel for the attribute
            return super().__getattr__(name)
        except AttributeError:
            # Obtain attribute from the wrapped model
            return getattr(self.module, name)


class CosineWarmup(torch.optim.lr_scheduler.LRScheduler):
    """
    A simple wrapper to implement reverse cosine annealing as a PyTorch LRScheduler.
    """

    def __init__(self, optimizer, n_steps):
        """
        Initialize the scheduler and set the duration of warmup.

        Parameters
        ----------
        optimizer : PyTorch Optimizer
          Optimizer object with learning rates to schedule
        n_steps : int
          Number of steps to reach maximum scaling
        """

        self.n_steps = max(0, n_steps)

        super().__init__(optimizer)

    def is_active(self):
        """
        Helper to determine when to stop stepping.
        """

        active = self.last_epoch < self.n_steps

        return active

    def reset(self):
        """
        Reset the scheduler.
        """

        self.last_epoch = -1
        self.step()

    def get_lr(self):
        """
        Obtain scheduler learning rates.
        """

        # Simply use closed form expression
        lr = self._get_closed_form_lr()

        return lr

    def _get_closed_form_lr(self):
        """
        Compute learning rates for the current step.
        """

        # Clamp current step at chosen number of steps
        curr_step = 1 + min(self.last_epoch, self.n_steps)
        # Compute scaling corresponding to current step
        scaling = 1 - 0.5 * (1 + cos(curr_step * pi / (self.n_steps + 1)))
        # Apply the scaling to each learning rate
        lr = [scaling * base_lr for base_lr in self.base_lrs]

        return lr


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
            # Update the tracked maximum gradient norm
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


class MultipitchEvaluator(object):
    """
    A simple tracker to store results and compute statistics across an entire test set.
    """

    def __init__(self, tolerance=0.5):
        """
        Initialize the tracker.

        Parameters
        ----------
        tolerance : float
          Semitone tolerance for correct predictions
        """

        self.tolerance = tolerance

        # Initialize dictionary to track results
        self.results = None
        self.reset_results()

    def reset_results(self):
        """
        Reset tracked results to empty dictionary.
        """

        self.results = {}

    def append_results(self, results):
        """
        Append the results for a test sample.

        Parameters
        ----------
        results : dict of {str : float} entries
          Numerical results for a single sample
        """

        # Loop through all keys
        for key in results.keys():
            if key in self.results.keys():
                # Add the score to the pre-existing array
                self.results[key] = np.append(self.results[key], results[key])
            else:
                # Initialize a new array for the metric
                self.results[key] = np.array([results[key]])

    def average_results(self):
        """
        Compute the mean and standard deviation for each metric across currently tracked results.

        Returns
        ----------
        mean : dict of {str : float} entries
          Average scores across currently tracked results
        std_dev : dict of {str : float} entries
          Standard deviation of scores across currently tracked results
        """

        # Clone all current scores
        mean = deepcopy(self.results)
        std_dev = deepcopy(self.results)

        # Loop through all metrics
        for key in self.results.keys():
            # Compute statistics for the metric
            mean[key] = round(np.mean(mean[key]), 5)
            std_dev[key] = round(np.std(std_dev[key]), 5)

        return mean, std_dev

    def evaluate(self, times_est, multi_pitch_est, times_ref, multi_pitch_ref):
        """
        Compute MPE results for a set of predictions using mir_eval.

        Parameters
        ----------
        times_est : ndarray (T)
          Times corresponding to multi-pitch estimates
        multi_pitch_est : list of ndarray (T x [...])
          Frame-level multi-pitch estimates
        times_ref : ndarray (K)
          Times corresponding to ground-truth multi-pitch
        multi_pitch_ref : list of ndarray (K x [...])
          Frame-level multi-pitch ground-truth

        Returns
        ----------
        results : dict of {str : float} entries
          Numerical MPE results for the predictions
        """

        # Use mir_eval to compute multi-pitch results at specified tolerance
        results = mir_eval.multipitch.evaluate(times_ref, multi_pitch_ref,
                                               times_est, multi_pitch_est,
                                               window=self.tolerance)

        # Make keys lowercase and switch to regular dict type
        results = {k.lower(): results[k] for k in results.keys()}

        # Extract precision and recall from results
        pr, rc = results['precision'], results['recall']

        # Compute f1-score as harmonic mean of precsion and recall
        f_measure = 2 * pr * rc / (pr + rc + sys.float_info.epsilon)

        # Add f1-score to mir_eval results
        results.update({'f1-score' : f_measure})

        for k in deepcopy(results).keys():
            # Prepend tag to indicate MPE metric
            results[f'mpe/{k}'] = results.pop(k)

        return results
