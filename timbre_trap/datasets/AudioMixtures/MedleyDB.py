from ..Common import MedleyDB

import os


class MedleyDB(MedleyDB):
    """
    Implements a wrapper for the MedleyDB dataset to analyze the full mixtures.
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Genres corresponding to the full song mixtures
        """

        splits = MedleyDB.available_genres()

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Genre identifier

        Returns
        ----------
        tracks : list of strings
          List containing the songs of the specified genre
        """

        # Initialize a list to hold valid tracks
        tracks = list()

        for multitrack in self.metadata.keys():
            # Check if multitrack has the specified genre
            if self.metadata[multitrack]['genre'] == split:
                # Add the multitrack name
                tracks.append(multitrack)

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MedleyDB track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Construct the path to the audio mixture
        wav_path = os.path.join(self.base_dir, 'Audio', track, f'{track}_MIX.wav')

        return wav_path
