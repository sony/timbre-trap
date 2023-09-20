from .. import MPEDataset
from ..Common import URMP

import numpy as np
import os


class URMP(MPEDataset, URMP):
    """
    Implements a wrapper for the URMP dataset to analyze the full mixtures.
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
          URMP track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Get the path to the audio mixture
        wav_path = os.path.join(self.base_dir, track, f'AuMix_{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the paths to a track's ground truth.

        Parameters
        ----------
        track : string
          URMP track name

        Returns
        ----------
        txt_paths : list of strings
          Path to ground-truth (multiple) for the specified track
        """

        # Obtain a list of all files under the track's directory
        track_files = os.listdir(os.path.join(self.base_dir, track))

        # Get the path for the F0 annotations of each instrument in the mixture
        txt_paths = [os.path.join(self.base_dir, track, f) for f in track_files
                      if f.startswith('F0s')]

        return txt_paths

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          URMP track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        pitches : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain the paths to the track's ground_truth
        txt_paths = self.get_ground_truth_path(track)

        # Initialize ground-truth variables
        times, pitches = None, None

        # Loop through annotations
        for t in txt_paths:
            # Read annotation file
            with open(t) as txt_file:
                # Read frame-level annotations into a list
                annotations = [f.split() for f in txt_file.readlines()]

            # Break apart frame time and pitch observations from annotations
            _times, _pitches = np.array(annotations).astype('float').T

            # Add a dimension to the pitches for instrument
            _pitches = np.expand_dims(_pitches, axis=0)

            if times is None or pitches is None:
                # Use provided times and pitches
                times, pitches, = _times, _pitches
            else:
                # Both sets of times should match
                assert np.allclose(times, _times)
                # Append the pitch observations
                pitches = np.concatenate((pitches, _pitches), axis=0)

        # Convert to pitch list representation
        pitches = [p[p != 0] for p in pitches.T]

        return times, pitches
