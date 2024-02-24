from ..Common import MusicNet

import pandas as pd


class MusicNet(MusicNet):
    """
    Implements a wrapper for the MusicNet dataset to analyze only the tracks with a single instrument class.
    """

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          String indicating train or test split

        Returns
        ----------
        tracks : list of strings
          Names of tracks with a single instrument class belonging to the split
        """

        # Obtain all tracks belonging to split
        tracks = super().get_tracks(split)

        # Loop though all tracks
        for t in tracks.copy():
            # Obtain the path to the track's ground_truth
            csv_path = self.get_ground_truth_path(t)

            # Load tabulated note data from the csv file
            note_entries = pd.read_csv(csv_path).to_numpy()

            # Unpack the instrument labels for the notes
            instruments = note_entries[:, 2].astype(int).tolist()

            if len(set(instruments)) > 1:
                # Discard track name
                tracks.remove(t)

        return tracks
