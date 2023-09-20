from datasets import NoteDataset, constants
from datasets.utils import threshold
from models.utils import filter_non_peaks
from models.objectives import *
from utils import *

from torchmetrics.audio import SignalDistortionRatio
from scipy.stats import hmean
from copy import deepcopy

import numpy as np
import mir_eval
import warnings
import librosa
import torch
import sys


EPSILON = sys.float_info.epsilon


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
                # Add the provided score to the pre-existing array
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
          Numerical MPE results for a set of predictions
        """

        # Use mir_eval to compute multi-pitch results at specified tolerance
        results = mir_eval.multipitch.evaluate(times_ref, multi_pitch_ref,
                                               times_est, multi_pitch_est,
                                               window=self.tolerance)

        # Make keys lowercase and switch to regular dict type
        results = {k.lower(): results[k] for k in results.keys()}

        # Calculate the f1-score using the harmonic mean formula
        f_measure = hmean([results['precision'] + EPSILON,
                           results['recall'] + EPSILON]) - EPSILON

        # Add f1-score to the mir_eval results
        results.update({'f1-score' : f_measure})

        for k in deepcopy(results).keys():
            # Prepend tag to indicate MPE metric
            results[f'mpe/{k}'] = results.pop(k)

        return results


def evaluate(model, eval_set, multipliers, writer=None, i=0, device='cpu'):
    # Initialize a new evaluator for the dataset
    evaluator = MultipitchEvaluator()

    # Add model to selected device and switch to evaluation mode
    model = model.to(device)
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
            # Extract audio and add to the appropriate device
            audio = data[constants.KEY_AUDIO].to(device).unsqueeze(1)
            # Extract ground-truth targets as a Tensor
            targets = torch.Tensor(data[constants.KEY_GROUND_TRUTH])

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

            # Perform transcription and reconstruction simultaneously
            reconstruction, latents, transcription, losses = model(audio)

            # Extract magnitude of decoded coefficients and convert to activations
            transcription = torch.nn.functional.tanh(model.sliCQ.to_magnitude(transcription))

            # Determine the times associated with predictions
            times_est = model.sliCQ.get_times(model.sliCQ.get_expected_frames(audio.size(-1)))
            # Perform peak-picking and thresholding on the activations
            activations = threshold(filter_non_peaks(to_array(transcription)), 0.5).squeeze(0)

            if np.sum(activations[valid_freqs]):
                # Print a warning message indicating invalid predictions
                warnings.warn('Positive activations were generated for '
                              'invalid frequencies.', RuntimeWarning)
                # Remove activations for invalid frequencies
                activations[valid_freqs] = 0

            # Convert the activations to frame-level multi-pitch estimates
            multi_pitch_est = eval_set.activations_to_multi_pitch(activations, model.sliCQ.midi_freqs)

            # Compute results for this track using mir_eval multi-pitch metrics
            results = evaluator.evaluate(times_est, multi_pitch_est, times_ref, multi_pitch_ref)
            # Store the computed results
            evaluator.append_results(results)

            # Invert reconstructed spectral coefficients to synthesize audio
            synth = model.sliCQ.decode(reconstruction)

            # Compute SDR w.r.t. original (padded) audio
            sdr = sdr_module(synth, audio).item()
            # Store the SDR for the track
            evaluator.append_results({'reconstruction/SDR' : sdr})

            # Obtain spectral coefficients of audio
            coefficients = model.sliCQ(audio)

            # Compute the reconstruction loss for the batch
            reconstruction_loss = compute_reconstruction_loss(reconstruction, coefficients)

            # Determine padding amount in terms of frames
            n_pad_frames = len(times_est) - targets.size(-1)

            # Pad the transcription targets to match prediction size
            targets = torch.nn.functional.pad(targets, (0, n_pad_frames))

            # Compute the transcription loss for the batch
            transcription_loss = compute_transcription_loss(transcription.squeeze(), targets.to(device), True)

            # Compute the total loss for the track
            total_loss = multipliers['reconstruction'] * reconstruction_loss + \
                         multipliers['transcription'] * transcription_loss

            for key_loss, val_loss in losses.items():
                # Store the model loss for the track
                evaluator.append_results({f'loss/{key_loss}' : val_loss.item()})
                # Add the model loss to the total loss
                total_loss += multipliers.get(key_loss, 1) * val_loss

            # Store all losses for the track
            evaluator.append_results({'loss/reconstruction' : reconstruction_loss.item(),
                                      'loss/transcription' : transcription_loss.item(),
                                      'loss/total' : total_loss.item()})

        # Compute the average for all scores
        average_results, _ = evaluator.average_results()

        if writer is not None:
            # Loop through all computed scores
            for key in average_results.keys():
                # Log the average score for this dataset
                writer.add_scalar(f'{eval_set.name()}/{key}', average_results[key], i)

            # Extract magnitude from original spectral coefficients and convert to decibels
            features_log = model.sliCQ.to_decibels(model.sliCQ.to_magnitude(coefficients))
            # Extract magnitude from reconstructed spectral coefficients and convert to decibels
            reconstruction = model.sliCQ.to_decibels(model.sliCQ.to_magnitude(reconstruction))

            # Add channel dimension to activations
            features_log = features_log.unsqueeze(-3)
            reconstruction = reconstruction.unsqueeze(-3)
            transcription = transcription.unsqueeze(-3)
            targets = targets.unsqueeze(-3)

            # Remove the batch dimension of input and outputs
            features_log = features_log.squeeze(0)
            reconstruction = reconstruction.squeeze(0)
            transcription = transcription.squeeze(0)

            # Reduce time resolution for better TensorBoard visualization
            features_log = torch.nn.functional.avg_pool2d(features_log, kernel_size=(1, 7))
            reconstruction = torch.nn.functional.avg_pool2d(reconstruction, kernel_size=(1, 7))
            transcription = torch.nn.functional.avg_pool2d(transcription, kernel_size=(1, 7))
            targets = torch.nn.functional.avg_pool2d(targets, kernel_size=(1, 7))

            # Visualize predictions for the final sample of the evaluation dataset
            writer.add_image(f'{eval_set.name()}/vis/original CQT', features_log.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/vis/recovered CQT', reconstruction.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/vis/ground-truth', targets.flip(-2), i)
            writer.add_image(f'{eval_set.name()}/vis/estimated', transcription.flip(-2), i)

    return average_results
