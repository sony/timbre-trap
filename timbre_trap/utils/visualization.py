from .processing import to_array

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import numpy as np


__all__ = [
    'initialize_figure',
    'plot_magnitude',
    'plot_latents'
]


def initialize_figure(figsize=(9, 3), interactive=False):
    """
    Create a new figure and display it.

    Parameters
    ----------
    figsize : tuple (x, y) or None (Optional)
      Size of plot window in inches
    interactive : bool
      Whether to turn on matplotlib interactive mode

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the created figure
    """

    if interactive and not plt.isinteractive():
        # Set pyplot to interactive mode
        plt.ion()

    # Create a new figure with the specified size
    fig = plt.figure(figsize=figsize, tight_layout=True)

    if not interactive:
        # Open figure manually
        plt.show(block=False)

    return fig


def plot_magnitude(magnitude, extent=None, colorbar=False, fig=None, save_path=None):
    """
    Plot magnitude coefficients within range [0, 1].

    Parameters
    ----------
    magnitude : ndarray (F x T)
      Magnitude coefficients [0, 1]
      F - number of frequency bins
      T - number of frames
    extent : list [l, r, b, t] or None (Optional)
      Boundaries of time and frequency axis
    colorbar : bool
      Whether to include a colorbar for reference
    fig : matplotlib Figure object
      Preexisting figure to use for plotting
    save_path : string or None (Optional)
      Save the figure to this path

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot TFR
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(interactive=False)

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    if extent is not None:
        # Swap position of bottom and top
        extent = [extent[0], extent[1], extent[3], extent[2]]

    # Display magnitude as an image heatmap
    img = ax.imshow(magnitude, vmin=0, vmax=1, extent=extent)
    # Flip y-axis for ascending pitch
    ax.invert_yaxis()
    # Make sure the image fills the figure
    ax.set_aspect('auto')

    if extent is not None:
        # Add axis labels
        ax.set_ylabel('Frequency (MIDI)')
        ax.set_xlabel('Time (s)')
    else:
        # Hide the axes
        ax.axis('off')

    if colorbar:
        # Include colorbar
        fig.colorbar(img)

    if save_path is not None:
        # Save figure to the specified path
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    return fig


def plot_latents(latents, labels, seed=0, fig=None, save_path=None):
    """
    Plot a set of latent codes with labels after reducing them to 2D representations.

    Parameters
    ----------
    latents : Tensor (L x D_lat)
      Collection of latent codes to visualize
    labels : list [*] * L
      Corresponding labels for the latent codes
    seed : int
      Seed for reproducibility
    fig : matplotlib Figure object
      Preexisting figure to use for plotting
    save_path : string or None (Optional)
      Save the figure to this path

    Returns
    ----------
    fig : matplotlib Figure object
      A handle for the figure used to plot the latent space visualization
    """

    if fig is None:
        # Initialize a new figure if one was not given
        fig = initialize_figure(figsize=(9, 6), interactive=False)

    # Obtain a handle for the figure's current axis
    ax = fig.gca()

    # Run TSNE to obtain a 2D representation of the latents
    latents_2d = TSNE(n_components=2,
                      perplexity=5,
                      n_iter=1000,
                      verbose=False,
                      random_state=seed).fit_transform(to_array(latents))

    # Represent labels as an array
    labels = np.array(labels)

    for l in np.unique(labels):
        # Determine which samples correspond to the label
        idcs = labels == l
        # Plot all 2D latents for the label
        ax.scatter(latents_2d[idcs, 0], latents_2d[idcs, 1], label=l, s=40)

    # Include legend for the latent labels
    ax.legend()

    # Move left y-axis and bottom x-axis to center
    ax.spines['left'].set_position('center')
    ax.spines['bottom'].set_position('center')

    # Eliminate upper and right axes
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')

    # Show ticks in left and lower axes only
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    for ticks_x in ax.xaxis.get_ticklabels()[::2]:
        # Remove every other tick on x-axis
        ticks_x.set_visible(False)

    for ticks_y in ax.yaxis.get_ticklabels()[::2]:
        # Remove every other tick on y-axis
        ticks_y.set_visible(False)

    # Add a title to the figure
    fig.suptitle('t-SNE Visualization of Latents Averaged Over Stems')

    if save_path is not None:
        # Save figure to the specified path
        fig.savefig(save_path, bbox_inches='tight', pad_inches=0)

    return fig
