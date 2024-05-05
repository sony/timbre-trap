import torch


__all__ = [
    'compute_reconstruction_loss',
    'compute_transcription_loss',
    'compute_consistency_loss'
]


def compute_reconstruction_loss(reconstructed, target):
    """
    Compute reconstruction loss for a batch.

    Parameters
    ----------
    reconstructed : Tensor (B x C_in x F X T)
      Batch of reconstructed coefficients
    target : Tensor (B x C_in x F X T)
      Batch of original coefficients

    Returns
    ----------
    reconstruction_loss : tensor (float)
      Total reconstruction loss for the batch
    """

    # Compute mean squared error with respect to every time-frequency bin of coefficients
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
      Whether to proportionally weight loss for positive targets

    Returns
    ----------
    transcription_loss : tensor (float)
      Total transcription loss for the batch
    """

    # Compute mean squared error between the estimated and ground-truth transcription
    transcription_loss = torch.nn.functional.mse_loss(estimate, target, reduction='none')

    if weight_positive_class:
        # Sum ground-truth and its complement for weights
        positive_weight = torch.sum(target, dim=-2, keepdim=True)
        negative_weight = torch.sum(1 - target, dim=-2, keepdim=True)
        # Compute multi-class imbalance ratio for each frame
        positive_scaling = negative_weight / (positive_weight + torch.finfo().eps)
        # Determine scaling for each loss element
        scaling = positive_scaling * (target == 1)
        # Correct scaling for negative activations
        scaling[scaling == 0] = 1
        # Scale transcription loss
        transcription_loss *= scaling

    # Sum transcription loss across frequency and average across time / batch
    transcription_loss = transcription_loss.sum(-2).mean()

    return transcription_loss


def compute_consistency_loss(spectral_coefficients, transcription_coefficients, target):
    """
    Compute consistency loss components (spectral | transcription) for a batch.

    Parameters
    ----------
    spectral_coefficients : Tensor (B x 2 x F X T)
      Batch of reconstruction spectral coefficients
    transcription_coefficients : Tensor (B x 2 x F X T)
      Batch of transcription spectral coefficients
    target : Tensor (B x 2 x F X T)
      Batch of target spectral coefficients

    Returns
    ----------
    consistency_spectral_loss : tensor (float)
      Total spectral-consistency loss for the batch
    consistency_score_loss : tensor (float)
      Total transcription-consistency loss for the batch
    """

    # Compute reconstruction loss between spectral coefficients and target coefficients
    consistency_spectral_loss = compute_reconstruction_loss(spectral_coefficients, target)

    # Compute reconstruction loss between transcription coefficients and target coefficients
    consistency_score_loss = compute_reconstruction_loss(transcription_coefficients, target)

    return consistency_spectral_loss, consistency_score_loss
