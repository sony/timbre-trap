import torch


__all__ = [
    'compute_reconstruction_loss',
    'compute_transcription_loss'
]


def compute_reconstruction_loss(reconstructed, target):
    """
    Compute reconstruction loss with respect to spectral features for a batch.

    Parameters
    ----------
    reconstructed : Tensor (B x C_in x F X T)
      Batch of reconstructed spectral features
    target : Tensor (B x C_in x F X T)
      Batch of original spectral features

    Returns
    ----------
    reconstruction_loss : tensor (float)
      Total reconstruction loss for the batch
    """

    # Compute mean squared error with respect to every time-frequency bin of spectral features
    reconstruction_loss = torch.nn.functional.mse_loss(reconstructed, target, reduction='none')
    # Sum reconstruction loss across channel / frequency and average across time / batch
    reconstruction_loss = reconstruction_loss.sum(-3).sum(-2).mean()

    return reconstruction_loss


def compute_transcription_loss(estimate, target, weight_positive_class=False):
    """
    Compute transcription loss for a batch of estimated scores.

    Parameters
    ----------
    estimate : Tensor (B x F x T)
      Batch of estimated scores
    target : Tensor (B x F x T)
      Batch of ground-truth scores
    weight_positive_class : bool
      Whether to apply weight to loss for positive targets

    Returns
    ----------
    transcription_loss : tensor (float)
      Total transcription loss for the batch
    """

    # Compute mean squared error between the estimated and ground-truth transcription
    transcription_loss = torch.nn.functional.mse_loss(estimate, target, reduction='none')

    if weight_positive_class:
        # Determine the inverse ratio between positive and negative activations
        positive_weight = torch.sum(1 - target).item() / torch.sum(target).item()
        # Scale transcription loss for positive targets
        transcription_loss[target == 1] *= positive_weight

    # Sum transcription loss across frequency and average across time / batch
    transcription_loss = transcription_loss.sum(-2).mean()

    return transcription_loss
