from timbre_trap.utils.processing import threshold, filter_non_peaks
from timbre_trap.utils.data import constants
from . import BaseDataset

from abc import abstractmethod

import numpy as np
import warnings
import librosa
import scipy


class PitchDataset(BaseDataset):
    """
    Implements functionality for a dataset with pitch annotations.
    """

    def __init__(self, cqt, resample_idcs=None, **kwargs):
        """
        Store the CQT module being used for parameter access.

        Parameters
        ----------
        cqt : CQT module wrapper
          Instantiated CQT feature extraction module
        resample_idcs : list of [int, int]
          Time index boundaries to use when resampling
        """

        BaseDataset.__init__(self, **kwargs)

        self.cqt = cqt

        if resample_idcs is None:
            # Default to boundaries
            resample_idcs = [0, -1]

        self.resample_idcs = resample_idcs

    @abstractmethod
    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          Track name

        Returns
        ----------
        ground_truth_path : string
          Path to ground-truth for the specified track
        """

        return NotImplementedError

    @abstractmethod
    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          Track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        pitches : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        return NotImplementedError

    def slice_times(self, times, n_frames=None, offset_t=None):
        """
        Slice frame times to a specified size.

        Parameters
        ----------
        times : ndarray (N)
          Time (seconds) associated with frames
        n_frames : int or None (Optional)
          Number of frames to slice
        offset_t : float or None (Optional)
          Offset (in seconds) for slice

        Returns
        ----------
        times : ndarray (M)
          Sliced frame times
        offset_n : float
          Offset (in frames) used to take slice
        """

        if n_frames is None:
            # Determine the expected sequence length
            n_samples = self.cqt.get_expected_samples(self.n_secs)
            # Determine the corresponding number of frames
            n_frames = self.cqt.get_expected_frames(n_samples)

        if len(times) >= n_frames:
            if offset_t is None:
                # Sample a starting frame index randomly for the excerpt
                start = self.rng.randint(0, times.size - n_frames + 1)
                # Track frame offset
                offset_n = start
                # Trim times to the sequence length
                times = times[start : start + n_frames]
            else:
                # Compute frame times for selected excerpt
                times = self.cqt.get_times(n_frames) + offset_t
                # Determine corresponding frame offset
                offset_n = offset_t * (self.cqt.sample_rate / self.cqt.hop_length)
        else:
            # Determine how much padding is required
            pad_total = n_frames - len(times)

            if offset_t is None:
                # Randomly distribute padding to both sides
                pad_left = self.rng.randint(0, pad_total)
            else:
                # Infer padding distribution from provided offset
                pad_left = round(abs(offset_t) * self.sample_rate / self.cqt.hop_length)

            # Track frame offset
            offset_n = -pad_left

            # Pad the times with -∞ and ∞ to indicate invalid times
            times = np.pad(times, (pad_left, 0), constant_values=-np.inf)
            times = np.pad(times, (0, pad_total - pad_left), constant_values=np.inf)

        return times, offset_n

    @abstractmethod
    def __getitem__(self, index, n_samples=None, offset_t=None):
        """
        Extract the ground-truth data for a sampled track.

        Parameters
        ----------
        index : int
          Index of sampled track
        n_samples : int
          Expected number of samples for the track
        offset_t : float or None (Optional)
          Offset (in seconds) for slice

        Returns
        ----------
        data : dict containing
          track : string
            Identifier for the track
          times : Tensor (T)
            Time associated with each frame
          ground_truth : Tensor (F x T)
            Ground-truth activations for the track
        """

        # Determine corresponding track
        track = self.tracks[index]

        # Read the track's multi-pitch annotations
        _times, _pitches = self.get_ground_truth(track)

        if n_samples is None:
            # Infer expected number of samples from ground-truth
            n_samples = self.cqt.get_expected_samples(_times[-1])

        # Determine expected number of frames and corresponding times
        times = self.cqt.get_times(self.cqt.get_expected_frames(n_samples))

        if self.n_secs is not None:
            # Randomly slice times using default sequence length
            times, _ = self.slice_times(times, offset_t=offset_t)

        # Obtain ground-truth resampled to sliced times
        multi_pitch = self.resample_multi_pitch(_times, _pitches, times)

        # Convert pitch annotations to multi pitch activations
        ground_truth = self.multi_pitch_to_activations(multi_pitch, self.cqt.get_midi_freqs())

        # Pack the data into a dictionary
        data = {constants.KEY_TRACK : track,
                constants.KEY_TIMES : times,
                constants.KEY_GROUND_TRUTH : ground_truth}

        return data

    def resample_multi_pitch(self, _times, _multi_pitch, times):
        """
        Resample ground-truth annotations to align with a new time grid.

        Parameters
        ----------
        _times : ndarray (T)
          Original times
        _multi_pitch : list of ndarray (T x [...])
          Multi-pitch annotations corresponding to original times
        times : ndarray (K)
          Target times for resampling

        Returns
        ----------
        multi_pitch : list of ndarray (K x [...])
          Multi-pitch annotations corresponding to target times
        """

        # Create an array of frame indices
        original_idcs = np.arange(len(_times))

        # Clamp resampled indices within the valid range
        fill_values = (original_idcs[self.resample_idcs[0]],
                       original_idcs[self.resample_idcs[-1]])

        # Obtain a function to resample annotation times
        res_func_time = scipy.interpolate.interp1d(x=_times,
                                                   y=original_idcs,
                                                   kind='nearest',
                                                   bounds_error=False,
                                                   fill_value=fill_values,
                                                   assume_sorted=True)

        # Resample the multi-pitch annotations using above interpolation function
        multi_pitch = [_multi_pitch[t] for t in res_func_time(times).astype('uint')]

        return multi_pitch

    @staticmethod
    def multi_pitch_to_activations(multi_pitch, midi_freqs, n_bins_blur=2):
        """
        Convert a sequence of active pitches into an array of discrete pitch activations.

        Parameters
        ----------
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        midi_freqs : ndarray (F)
          MIDI frequency corresponding to each bin
        n_bins_blur : int
          Length about center of blurring kernel

        Returns
        ----------
        activations : ndarray (F x T)
          Discrete activations corresponding to MIDI pitches
        """

        # Obtain a function to resample to discrete frequencies
        res_func_freq = scipy.interpolate.interp1d(x=midi_freqs,
                                                   y=np.arange(len(midi_freqs)),
                                                   kind='nearest',
                                                   bounds_error=True,
                                                   assume_sorted=True)

        # Construct an empty array of appropriate size
        activations = np.zeros((len(midi_freqs), len(multi_pitch)))

        # Make sure zeros are filtered out and convert pitches to MIDI
        multi_pitch = [librosa.hz_to_midi(p[p != 0]) for p in multi_pitch]

        # Count the number of nonzero pitch annotations
        num_nonzero = sum([sum(a != 0) for a in multi_pitch])

        # Determine lower and upper pitch boundaries
        lb, ub = np.min(midi_freqs), np.max(midi_freqs)

        # Filter out any out-of-bounds pitches from the annotations
        multi_pitch = [p[np.logical_and(p >= lb, p <= ub)] for p in multi_pitch]

        # Count the number of valid pitch annotations
        num_valid = sum([sum(a != 0) for a in multi_pitch])

        if num_valid != num_nonzero:
            # Print a warning message indicating incomplete ground-truth
            warnings.warn('Could not fully represent ground-truth with '
                          'available frequency bins.', RuntimeWarning)

        if num_valid:
            # Obtain frame indices corresponding to pitch activity
            frame_idcs = np.concatenate([[i] * len(multi_pitch[i])
                                         for i in range(len(multi_pitch)) if len(multi_pitch[i])])

            # Determine the closest frequency bin for each pitch observation
            multi_pitch_idcs = np.concatenate([res_func_freq(multi_pitch[i])
                                               for i in sorted(set(frame_idcs))]).astype('int')

            if n_bins_blur:
                # Create relative bin indices for kernel
                bin_idcs = np.arange(1 + 2 * n_bins_blur) - n_bins_blur
                # Create Gaussian blur kernel with unit standard deviation
                kernel = np.exp(-(1 / 2) * (bin_idcs / 1) ** 2)

                # TODO - stronger blur for overlapping activations?

                # Iterate through positive activations
                for i, j in zip(multi_pitch_idcs, frame_idcs):
                    # Obtain absolute indices for blurred activations
                    start_i, end_i = i - n_bins_blur, i + n_bins_blur + 1

                    # Clip blurred indices according to activation boundaries
                    start_clip, end_clip = max(0, start_i), min(len(midi_freqs), end_i)

                    # Determine corresponding indices relative to kernel
                    start_rel = abs(min(0, start_i))
                    end_rel = len(kernel) - abs(min(0, len(midi_freqs) - end_i))

                    # Insert values without superimposing kernels for multiple activations
                    activations[start_clip : end_clip, j] = np.maximum(kernel[start_rel : end_rel],
                                                                       activations[start_clip : end_clip, j])
            else:
                # Insert activations into the ground-truth array
                activations[multi_pitch_idcs, frame_idcs] = 1

        return activations

    @staticmethod
    def activations_to_multi_pitch(activations, midi_freqs, peaks_only=False, t=0.5):
        """
        Convert an array of discrete pitch activations into a sequence of active pitches.

        Parameters
        ----------
        activations : ndarray (F x T)
          Binarized activations corresponding to midi_freqs
        midi_freqs : ndarray (F)
          MIDI frequency corresponding to each bin
        peaks_only : bool
          Whether to perform local peak-picking
        t : float [0, 1]
          Threshold value to binarize input

        Returns
        ----------
        multi_pitch : list of ndarray (T x [...])
          Array of active pitches (in Hertz) across time
        """

        # Initialize empty pitch arrays for each frame
        multi_pitch = [np.empty(0)] * activations.shape[-1]

        if peaks_only:
            # Remove non-local-peaks along frequency
            activations = filter_non_peaks(activations)

        # Binarize activations using threshold
        activations = threshold(activations, t)

        # Determine which frames contain pitch activity
        non_silent_frames = np.where(np.sum(activations, axis=-2) > 0)[-1]

        # Loop through non-silent frames
        for i in list(non_silent_frames):
            # Determine the active pitches within frame and insert into multi-pitch list
            multi_pitch[i] = librosa.midi_to_hz(midi_freqs[np.where(activations[..., i])[-1]])

        return multi_pitch
