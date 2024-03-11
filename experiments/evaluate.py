from timbre_trap.datasets import NoteDataset
from timbre_trap.framework import *
from timbre_trap.utils import *

from torchmetrics.audio import SignalDistortionRatio

import numpy as np
import mir_eval
import warnings
import librosa
import torch


def evaluate(model, eval_set, multipliers, writer=None, i=0, device='cpu'):
    """
    Evaluate a model on a given dataset.

    Parameters
    ----------
    model : TimbreTrap
      Model checkpoint to evaluate
    eval_set : MPEDataset or AMTDataset
      Dataset with audio and ground-truth
    multipliers : list of float
      Scaling factor for each loss term
    writer : TensorboardWriter
      Tensorboard writer to use for logging
    i : int
      Checkpoint number to use for logging
    device : int or torch.device
      Device to use for evaluation

    Returns
    ----------
    average_Results : dict of {string : float}
      Average results across the dataset for each metric
    """

    # Initialize a clean evaluator
    evaluator = MultipitchEvaluator()

    # Add model to selected device
    model = model.to(device)
    # Switch to evaluation mode
    model.eval()

    # Determine valid frequency bins for multi-pitch estimation (based on mir_eval)
    valid_freqs = librosa.midi_to_hz(model.sliCQ.midi_freqs) > mir_eval.multipitch.MAX_FREQ

    # Initialize a module to compute SDR
    sdr_module = SignalDistortionRatio().to(device)

    with torch.no_grad():
        # Loop through tracks
        for data in eval_set:
            # Determine which track is being processed
            track = data[constants.KEY_TRACK]
            # Extract audio and add to appropriate device
            audio = data[constants.KEY_AUDIO].to(device)
            # Extract ground-truth targets
            targets = data[constants.KEY_GROUND_TRUTH]
            # Convert to Tensor and add to appropriate device
            targets = torch.Tensor(targets).to(device)
            # Add batch dimension to audio and ground-truth
            audio, targets = audio.unsqueeze(0), targets.unsqueeze(0)

            if isinstance(eval_set, NoteDataset):
                # Extract frame times of ground-truth targets as reference
                times_ref = data[constants.KEY_TIMES]
                # Obtain the ground-truth note annotations
                pitches, intervals = eval_set.get_ground_truth(track)
                # Convert note pitches to Hertz
                pitches = librosa.midi_to_hz(pitches)
                # Convert the note annotations to multi-pitch annotations
                multi_pitch_ref = eval_set.notes_to_multi_pitch(pitches, intervals, times_ref)
            else:
                # Obtain the ground-truth multi-pitch annotations
                times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)

            # Pad audio to next multiple of block length
            audio = model.sliCQ.pad_to_block_length(audio)

            # Obtain spectral coefficients of audio
            coefficients = model.sliCQ(audio)

            if isinstance(model, TimbreTrapMag):
                # Take magnitude of complex coefficients for reconstruction loss
                coefficients = model.sliCQ.to_magnitude(coefficients).unsqueeze(-3)
            if isinstance(model, TimbreTrapMagDB):
                # Convert amplitude to rescaled decibels
                coefficients = model.sliCQ.to_decibels(coefficients)

            # Perform transcription and reconstruction simultaneously
            reconstruction, latents, transcription_coeffs, \
            transcription_rec, transcription_scr, losses = model(audio, multipliers['consistency'])

            # Convert transcription coefficients to activations
            transcription = model.to_activations(transcription_coeffs)
            # Remove batch dimension and convert to array
            activations = to_array(transcription.squeeze(0))

            # Determine the times associated with predictions
            times_est = model.sliCQ.get_times(model.sliCQ.get_expected_frames(audio.size(-1)))

            if np.sum(activations[valid_freqs]):
                # Print a warning message indicating invalid predictions
                warnings.warn('Positive activations were generated for '
                              'invalid frequencies.', RuntimeWarning)
                # Remove activations for invalid frequencies
                activations[valid_freqs] = 0

            # Convert the activations to frame-level multi-pitch estimates
            multi_pitch_est = eval_set.activations_to_multi_pitch(activations, model.sliCQ.midi_freqs, peaks_only=True)

            # Compute results for the track using mir_eval multi-pitch metrics
            results = evaluator.evaluate(times_est, multi_pitch_est, times_ref, multi_pitch_ref)
            # Store the computed results
            evaluator.append_results(results)

            if not isinstance(model, TimbreTrapMag):
                # Synthesize audio from reconstructed spectral coefficients
                synth = model.sliCQ.decode(reconstruction)

                # Compute SDR w.r.t. original (padded) audio
                sdr = sdr_module(synth, audio).item()
                # Store the SDR for the track
                evaluator.append_results({'reconstruction/SDR' : sdr})

            # Compute the reconstruction loss for the track
            reconstruction_loss = compute_reconstruction_loss(reconstruction, coefficients)

            # Determine padding amount in frames
            n_pad_frames = len(times_est) - targets.size(-1)

            # Pad the transcription targets to match prediction size
            targets = torch.nn.functional.pad(targets, (0, n_pad_frames))

            # Compute the transcription loss for the track
            transcription_loss = compute_transcription_loss(transcription, targets, True)

            # Compute the total loss for the track
            total_loss = multipliers['reconstruction'] * reconstruction_loss + \
                         multipliers['transcription'] * transcription_loss

            if multipliers['consistency']:
                # Compute the total consistency loss for the track
                consistency_loss = sum(compute_consistency_loss(transcription_rec,
                                                                transcription_scr,
                                                                transcription_coeffs))
                # Store the consistency loss for the track
                evaluator.append_results({'loss/consistency' : consistency_loss.item()})
                # Add combined consistency loss to the total loss
                total_loss += multipliers['consistency'] * consistency_loss

            for key_loss, val_loss in losses.items():
                # Store the model loss for the track
                evaluator.append_results({f'loss/{key_loss}' : val_loss.item()})
                # Add the model loss to the total loss
                total_loss += multipliers.get(key_loss, 1) * val_loss

            # Store all losses for the track
            evaluator.append_results({'loss/reconstruction' : reconstruction_loss.item(),
                                      'loss/transcription' : transcription_loss.item(),
                                      'loss/total' : total_loss.item()})

        # Average results over all tracks
        average_results, _ = evaluator.average_results()

        if writer is not None:
            # Loop through each metric
            for key in average_results.keys():
                # Log the average score for this dataset
                writer.add_scalar(f'{eval_set.name()}/{key}', average_results[key], i)

            # Take magnitude of original spectral coefficients and convert to decibels
            features_log = model.sliCQ.to_decibels(model.sliCQ.to_magnitude(coefficients))
            # Take magnitude of reconstructed spectral coefficients and convert to decibels
            reconstruction = model.sliCQ.to_decibels(model.sliCQ.to_magnitude(reconstruction))

            # Reduce time resolution for better TensorBoard visualization
            features_log = torch.nn.functional.avg_pool2d(features_log, kernel_size=(1, 7))
            reconstruction = torch.nn.functional.avg_pool2d(reconstruction, kernel_size=(1, 7))
            transcription = torch.nn.functional.avg_pool2d(transcription, kernel_size=(1, 7))
            targets = torch.nn.functional.avg_pool2d(targets, kernel_size=(1, 7))

            # Visualize predictions for the final sample of the evaluation dataset
            writer.add_image(f'{eval_set.name()}/vis/magnitude CQT', features_log.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/vis/reconstruction', reconstruction.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/vis/ground-truth', targets.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/vis/transcription', transcription.flip(-2), i)

    return average_results
