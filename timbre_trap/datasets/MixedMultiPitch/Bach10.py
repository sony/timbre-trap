from .. import MPEDataset
from ..Common import Bach10

import numpy as np
import librosa
import scipy
import os


class Bach10(MPEDataset, Bach10):
    """
    Implements a wrapper for the Bach10 dataset to analyze the full mixtures.
    """

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Individual track (mixture) index

        Returns
        ----------
        tracks : list of strings
          List containing the individual track specified
        """

        # Obtain the full name corresponding to the track index
        tracks = [d for d in os.listdir(self.base_dir) if d.startswith(split)]

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Construct the path to the audio mixture
        wav_path = os.path.join(self.base_dir, f'{track}', f'{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        mat_path : string
          Path to ground-truth for the specified track
        """

        # Construct the path to the F0 annotations for the whole mixture
        mat_path = os.path.join(self.base_dir, f'{track}', f'{track}-GTF0s.mat')

        return mat_path

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          Bach10 track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        pitches : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain the path to the track's ground_truth
        mat_path = self.get_ground_truth_path(track)

        # Extract the frame-level multi-pitch annotations
        multi_pitch = scipy.io.loadmat(mat_path)['GTF0s']

        # Determine how many frames were provided
        num_frames = multi_pitch.shape[-1]

        # Compute the original times for each frame
        times = 0.023 + 0.010 * np.arange(num_frames)

        # Obtain ground-truth as a list of pitch observations in Hertz
        pitches = [librosa.midi_to_hz(p[p != 0]) for p in multi_pitch.T]

        return times, pitches
