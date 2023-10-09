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

    def get_ground_truth_path(self, track, instrument):
        """
        Get the path to a track's ground-truth for a specific instrument.

        Parameters
        ----------
        track : string
          URMP track name
        instrument : int
          Instrument index within track

        Returns
        ----------
        txt_path : string
          Path to ground-truth for the specified track and instrument
        """

        # Construct a path to the track directory
        track_dir = os.path.join(self.base_dir, track)

        # Identify the annotations file corresponding to the specified instrument index
        anno_file = [f for f in os.listdir(track_dir) if f.startswith(f'F0s_{instrument}')][0]

        # Construct a path to annotations for the instrument
        txt_path = os.path.join(track_dir, anno_file)

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

        # Obtain indices for instruments within mixture
        instruments = [(i + 1) for i in range(len(track.split('_')[2:]))]

        # Initialize ground-truth variables
        times, pitches = None, None

        # Loop through instruments
        for instrument in instruments:
            # Obtain the path to the ground_truth for the instrument
            txt_path = self.get_ground_truth_path(track, instrument)

            # Read pitch annotations
            with open(txt_path) as txt_file:
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
