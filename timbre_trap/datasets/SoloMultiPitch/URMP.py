from .. import MPEDataset
from ..Common import URMP

import numpy as np
import os


class URMP(MPEDataset, URMP):
    """
    Implements a wrapper for the URMP dataset to analyze the stems individually.
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
          List containing the stems of the track specified
        """

        # Determine the full name corresponding to the track index
        name = [d for d in os.listdir(self.base_dir) if d.startswith(split)][0]

        # Obtain a list of all files under the track's directory
        track_files = os.listdir(os.path.join(self.base_dir, name))

        # Get the names of the individual stem files for the track
        tracks = [os.path.join(name, f) for f in track_files if f.startswith('AuSep')]

        # Remove prefix and extension from each track name
        tracks = [os.path.splitext(t)[0].replace('AuSep_', '') for t in tracks]

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

        # Break apart the track name
        mixture, stem = os.path.split(track)

        # Construct the path to the audio stem
        wav_path = os.path.join(self.base_dir, mixture, f'AuSep_{stem}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          URMP track name

        Returns
        ----------
        txt_path : string
          Path to ground-truth for the specified track
        """

        # Break apart the track name
        mixture, stem = os.path.split(track)

        # Construct the path to the F0 annotations for the stem
        txt_path = os.path.join(self.base_dir, mixture, f'F0s_{stem}.txt')

        return txt_path

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

        # Obtain the path to the track's ground_truth
        txt_path = self.get_ground_truth_path(track)

        # Read annotation file
        with open(txt_path) as txt_file:
            # Read frame-level annotations into a list
            annotations = [f.split() for f in txt_file.readlines()]

        # Break apart frame time and pitch observations from annotations
        times, pitches = np.array(annotations).astype('float').T

        # Add a dimension to the pitches for instrument
        pitches = np.expand_dims(pitches, axis=-1)

        # Convert to pitch list representation
        pitches = [p[p != 0] for p in pitches]

        return times, pitches
