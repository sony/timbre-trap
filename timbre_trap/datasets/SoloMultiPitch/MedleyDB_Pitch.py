from timbre_trap.utils.data import constants
from .. import MPEDataset
from ..Common import MedleyDB

import pandas as pd
import json
import os


class MedleyDB_Pitch(MPEDataset, MedleyDB):
    """
    Implements a wrapper for the MedleyDB Pitch Tracking Subset to analyze the stems.
    """

    def __init__(self, **kwargs):
        """
        Override the resampling indices to prevent erroneous extension of pitches.
        """

        # Determine if keyword arguments were provided
        resample_idcs = kwargs.pop('resample_idcs', None)
        base_dir = kwargs.pop('base_dir', None)

        if resample_idcs is None:
            # Change the default resampling indices
            resample_idcs = [0, 0]

        if base_dir is None:
            # Assume the dataset exists at the default location
            base_dir = os.path.join(constants.DEFAULT_LOCATION, self.name())

        self.base_dir = base_dir

        # Update the values within the keyword arguments
        kwargs.update({'resample_idcs' : resample_idcs})
        kwargs.update({'base_dir' : self.base_dir})

        # Create dictionary for all metadata
        self.metadata = None
        self.load_metadata()

        MPEDataset.__init__(self, **kwargs)

    def load_metadata(self):
        """
        Load and process all metadata.
        """

        # Construct the path to the JSON metadata for MedleyDB-Pitch
        json_path = os.path.join(self.base_dir, 'medleydb_pitch_metadata.json')

        with open(json_path) as f:
            # Read JSON metadata
            self.metadata = json.load(f)

    @classmethod
    def name(cls):
        """
        Obtain a string representing the dataset.

        Returns
        ----------
        name : string
          Dataset name with dashes
        """

        # Obtain class name and replace underscores with dashes
        tag = super().name().replace('_', '-')

        return tag

    def available_multitracks(self):
        """
        Override parent function to indicate there are no multitracks.
        """

        return NotImplementedError

    @staticmethod
    def available_instruments():
        """
        Obtain a list of instruments in the dataset.

        See https://github.com/marl/medleydb/blob/master/medleydb/resources/taxonomy.yaml

        Returns
        ----------
        instruments : list of strings
          Instruments played within stems
        """

        instruments = [
        # strings - bowed
            'erhu',
            'violin',
            'viola',
            'cello',
            'double bass',
        # strings - plucked
            # None
        # strings - struck
            # None
        # winds - flutes
            'dizi',
            'flute',
            'piccolo',
            'bamboo flute',
        # winds - single reeds
            'alto saxophone',
            'baritone saxophone',
            'bass clarinet',
            'clarinet',
            'tenor saxophone',
            'soprano saxophone',
        # winds - double reeds
            'oboe',
        # winds - brass
            'trumpet',
            'french horn',
        # winds - free reeds
            # None
        # voices
            'male singer',
            'female singer',
        # percussion - idiophones
            # None
        # percussion - drums
            # None
        # electric - amplified
            'electric bass',
        # electric - electronic
            # None
        # other
            # None
        ]

        return instruments

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Solo instruments within the collection of stems
        """

        splits = MedleyDB_Pitch.available_instruments()

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Instrument identifier

        Returns
        ----------
        tracks : list of strings
          List containing the stems with the specified instrument
        """

        # Collect track names with the specified instrument in the metadata
        tracks = [t for t in self.metadata.keys() if split in self.metadata[t]['instrument']]

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MedleyDB-Pitch track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Construct the path to the audio stem
        wav_path = os.path.join(self.base_dir, 'audio', f'{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          MedleyDB-Pitch track name

        Returns
        ----------
        csv_path : string
          Path to ground-truth for the specified track
        """

        # Construct the path to the F0 annotations for the stem
        csv_path = os.path.join(self.base_dir, 'pitch', f'{track}.csv')

        return csv_path

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          MedleyDB-Pitch track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        pitches : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain the path to the track's ground_truth
        csv_path = self.get_ground_truth_path(track)

        # Load tabulated pitch data from the csv file and unpack it
        times, pitches = pd.read_csv(csv_path, header=None).to_numpy().T

        # Convert to pitch list representation
        pitches = [p[p != 0] for p in pitches]

        return times, pitches
