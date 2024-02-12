from . import PitchDataset, constants

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

        # Read the track's note annotations
        pitches, intervals = self.get_ground_truth(track)

        # Convert note pitches to Hertz
        pitches = librosa.midi_to_hz(pitches)

        # Determine frame times given the expected number of frames within amount of time defined by annotations
        times = self.cqt.get_times(self.cqt.get_expected_frames(self.cqt.get_expected_samples(np.max(intervals))))

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
                # Pad the times with -∞ and ∞ to indicate invalid times
                times = np.pad(times, (pad_left, 0), constant_values=-np.inf)
                times = np.pad(times, (0, pad_total - pad_left), constant_values=np.inf)

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
                # Add a pitch observation during the frame
                multi_pitch[i] = np.append(multi_pitch[i], p)

        return multi_pitch
