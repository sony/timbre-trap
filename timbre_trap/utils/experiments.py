import random
import torch
import math


__all__ = [
    'seed_everything',
    'print_and_log',
    'DataParallel',
    'CosineWarmup',
    'sum_gradient_norms',
    'average_gradient_norms',
    'get_max_gradient',
    'get_max_gradient_norm',
    'log_gradient_norms'
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
    path : str (None to bypass)
      Path to file to write text
    """

    # Print text to the console
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
            # Check DataParallel
            return super().__getattr__(name)
        except AttributeError:
            # Check the wrapped model
            return getattr(self.module, name)


class CosineWarmup(torch.optim.lr_scheduler.LRScheduler):
    """
    A simple wrapper to implement reverse cosine annealing as a PyTorch LRScheduler.
    """

    def __init__(self, optimizer, n_steps, last_epoch=-1, verbose=False):
        """
        Initialize the scheduler and set the duration of warmup.

        Parameters
        ----------
        See LRScheduler class...
        """

        self.n_steps = max(1, n_steps)

        super().__init__(optimizer, last_epoch, verbose)

    def is_active(self):
        """
        Helper to determine when to stop stepping.
        """

        active = self.last_epoch < self.n_steps

        return active

    def get_lr(self):
        """
        Obtain scheduler learning rates.
        """

        # Simply use closed form expression
        lr = self._get_closed_form_lr()

        return lr

    def _get_closed_form_lr(self):
        """
        Compute the learning rates for the current step.
        """

        # Clamp the current step at the chosen number of steps
        curr_step = max(0, min(self.last_epoch, self.n_steps))
        # Compute scaling corresponding to current step
        scaling = 1 - 0.5 * (1 + math.cos(curr_step * math.pi / self.n_steps))
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
