from . import BaseDataset, constants

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
        cqt : CQT
          Instantiated CQT feature extraction module
        resample_idcs : list of [int, int]
          Time index boundaries to use when resampling
        """

        BaseDataset.__init__(self, **kwargs)

        assert self.sample_rate == cqt.sample_rate

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

    @abstractmethod
    def __getitem__(self, index):
        """
        Extract the ground-truth data for a sampled track.

        Parameters
        ----------
        index : int
          Index of sampled track

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

        # Determine frame times given the expected number of frames within amount of time defined by annotations
        times = self.cqt.get_times(self.cqt.get_expected_frames(self.cqt.get_expected_samples(_times[-1])))

        if self.n_secs is not None:
            # Determine the required sequence length
            n_samples = self.cqt.get_expected_samples(self.n_secs)
            # Determine the corresponding number of frames
            n_frames = self.cqt.get_expected_frames(n_samples)

            if times.size >= n_frames:
                # Sample a random starting index for the trim
                start = self.rng.randint(0, times.size - n_frames + 1)
                # Obtain the stopping index
                stop = start + n_frames
                # Trim times to frame length
                times = times[start : stop]
            else:
                # Determine how much padding is required
                pad_total = n_frames - times.size
                # Randomly distribute between both sides
                pad_left = self.rng.randint(0, pad_total)
                # Pad the times with -1 to indicate invalid times
                # TODO - verify that resampling after padding won't cause problems...
                times = np.pad(times, (pad_left, pad_total - pad_left), constant_values=-1)

        # Obtain ground-truth resampled to computed times
        multi_pitch = self.resample_multi_pitch(_times, _pitches, times)

        # Convert pitch annotations to multi pitch activations
        ground_truth = self.multi_pitch_to_activations(multi_pitch, self.cqt.midi_freqs)

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
    def multi_pitch_to_activations(multi_pitch, midi_freqs):
        """
        Convert a sequence of active pitches into an array of discrete pitch activations.

        Parameters
        ----------
        multi_pitch : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        midi_freqs : ndarray (F)
          MIDI frequency corresponding to each bin

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

        # Construct an empty array of appropriate size by default
        activations = np.zeros((len(midi_freqs), len(multi_pitch)))

        # Make sure zeros are filtered out and convert to MIDI
        multi_pitch = [librosa.hz_to_midi(p[p != 0]) for p in multi_pitch]

        # Count the number of nonzero pitch annotations
        num_nonzero = sum([sum(a != 0) for a in multi_pitch])

        # Determine the lower and upper pitch boundaries
        lb, ub = np.min(midi_freqs), np.max(midi_freqs)

        # Filter out any out-of-bounds pitches from the annotations
        multi_pitch = [p[np.logical_and(p >= lb, p <= ub)] for p in multi_pitch]

        # Count the number of valid pitch annotations
        num_valid = sum([sum(a != 0) for a in multi_pitch])

        if num_valid != num_nonzero:
            # Print a warning message indicating degraded ground-truth
            warnings.warn('Could not fully represent ground-truth with '
                          'available frequency bins.', RuntimeWarning)

        if num_valid:
            # Obtain frame indices corresponding to pitch activity
            frame_idcs = np.concatenate([[i] * len(multi_pitch[i])
                                         for i in range(len(multi_pitch)) if len(multi_pitch[i])])

            # Determine the closest frequency bin for each pitch observation
            multi_pitch_idcs = np.concatenate([res_func_freq(multi_pitch[i])
                                               for i in sorted(set(frame_idcs))])

            # Insert pitch activity into the ground-truth
            activations[multi_pitch_idcs.astype('uint'), frame_idcs] = 1

        return activations

    @staticmethod
    def activations_to_multi_pitch(activations, midi_freqs, thr=0.5):
        """
        Convert an array of discrete pitch activations into a sequence of active pitches.

        Parameters
        ----------
        activations : ndarray (F x T)
          Binarized activations corresponding to midi_freqs
        midi_freqs : ndarray (F)
          MIDI frequency corresponding to each bin
        thr : float [0, 1]
          Threshold value

        Returns
        ----------
        multi_pitch : list of ndarray (T x [...])
          Array of active pitches (in Hertz) across time
        """

        # Initialize empty pitch arrays for each frame
        multi_pitch = [np.empty(0)] * activations.shape[-1]

        # Make sure provided activations are binarized
        assert np.alltrue(np.logical_or(activations == 0, activations == 1))

        # Determine which frames contain pitch activity
        non_silent_frames = np.where(np.sum(activations, axis=-2) > 0)[-1]

        # Loop through these frames
        for i in list(non_silent_frames):
            # Determine the active pitches within the frame and insert into the list
            multi_pitch[i] = librosa.midi_to_hz(midi_freqs[np.where(activations[..., i])[-1]])

        return multi_pitch
