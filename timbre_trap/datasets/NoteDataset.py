from timbre_trap.utils.data import constants
from . import PitchDataset

from abc import abstractmethod

import numpy as np
import librosa


class NoteDataset(PitchDataset):
    """
    Implements functionality for a dataset with note annotations.
    """

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
        pitches : ndarray (L)
          Array of note pitches
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        """

        return NotImplementedError

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

        # Read the track's note annotations
        pitches, intervals = self.get_ground_truth(track)

        # Convert note pitches to Hertz
        pitches = librosa.midi_to_hz(pitches)

        if n_samples is None:
            # Infer expected number of samples from ground-truth
            n_samples = self.cqt.get_expected_samples(np.max(intervals))

        # Determine expected number of frames and corresponding times
        times = self.cqt.get_times(self.cqt.get_expected_frames(n_samples))

        if self.n_secs is not None:
            # Randomly slice times using default sequence length
            times, _ = self.slice_times(times, offset_t=offset_t)

        # Convert note annotations to multi pitch annotations
        multi_pitch = self.notes_to_multi_pitch(pitches, intervals, times)

        # Convert pitch list observations to multi pitch activations
        ground_truth = self.multi_pitch_to_activations(multi_pitch, self.cqt.get_midi_freqs())

        # Pack the data into a dictionary
        data = {constants.KEY_TRACK : track,
                constants.KEY_TIMES : times,
                constants.KEY_GROUND_TRUTH : ground_truth}

        return data

    @staticmethod
    def notes_to_multi_pitch(pitches, intervals, times):
        """
        Convert a collection of pitch-interval pairs (notes) into a sequence of active pitches.

        Parameters
        ----------
        pitches : ndarray (L)
          Array of note pitches
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        times : ndarray (N)
          Target times for resampling

        Returns
        ----------
        multi_pitch : list of ndarray (N x [...])
          Array of active pitches (in Hertz) across time
        """

        # Initialize empty pitch arrays for each frame
        multi_pitch = [np.empty(0)] * times.shape[-1]

        # Loop through pitch-interval pairs
        for p, (j, k) in zip(pitches, intervals):
            # Loop through time indices corresponding to note activity
            for i in np.where((times >= j) & (times < k))[0]:
                # Insert pitch observation into the frame entry
                multi_pitch[i] = np.append(multi_pitch[i], p)

        return multi_pitch
