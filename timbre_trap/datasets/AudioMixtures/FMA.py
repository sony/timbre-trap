from timbre_trap.utils.data import *
from .. import AudioDataset

import pandas as pd
import os


class FMA(AudioDataset):
    """
    Implements a wrapper for the Free Music Archive dataset (https://github.com/mdeff/fma).
    """

    SIZE = None

    def __init__(self, **kwargs):
        """
        Add a field to store metadata for all available multitracks.
        """

        self.metadata = None

        AudioDataset.__init__(self, **kwargs)

    def load_metadata(self):
        """
        Load and process relevant metadata for splits.
        """

        # Construct the path to the list of genres in the dataset
        genres_path = os.path.join(self.base_dir, 'fma_metadata', 'genres.csv')

        # Load tabulated genre data from the csv file
        genre_entries = pd.read_csv(genres_path)

        # Extract the genre IDs, names, and parents
        genre_ids = list(genre_entries.genre_id)
        sub_genres = list(genre_entries.title)
        root_genres = list(genre_entries.top_level)
        # Convert root genre indices to strings
        root_genres = [sub_genres[genre_ids.index(k)] for k in root_genres]

        # Create a lookup table from sub-genre to root genre
        genre_lookup = dict(zip(genre_ids, root_genres))

        # Construct the path to the list of tracks in the dataset
        tracks_path = os.path.join(self.base_dir, 'fma_metadata', 'tracks.csv')

        # Load tabulated track data from the csv file
        track_entries = pd.read_csv(tracks_path, skiprows=2)

        # Extract track IDs from the data
        track_ids = list(track_entries.track_id)
        # Extract genre entries (track.genres) from the data
        track_genres = list(track_entries.pop('Unnamed: 41'))

        # Loop through the genre entries for each track
        for i, genres in enumerate(track_genres.copy()):
            if len(genres) > 2:
                # Convert genre entry to a list
                genres = genres[1:-1].split(', ')
                # Use the lookup table to obtain names for sub-genres
                genres = [genre_lookup[int(k)] for k in genres]
                # Remove redunant genres
                track_genres[i] = list(set(genres))
            else:
                # Insert a null genre
                track_genres[i] = ['None']

        # Convert track IDs to appropriate naming scheme
        track_ids = [f'{int(t):06d}' for t in track_ids]

        # Populate the metadata with tracks and genres
        self.metadata = dict(zip(track_ids, track_genres))

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Top-level genre categories
        """

        # All parent genres in the dataset (in order of occurrences)
        splits = ['Rock', 'Electronic', 'Experimental', 'Hip-Hop', 'Folk', 'Instrumental',
                  'Pop', 'International', 'Classical', 'Old-Time / Historic', 'Jazz',
                  'Country', 'Soul-RnB', 'Spoken', 'Blues', 'Easy Listening', 'None']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Top-level directory

        Returns
        ----------
        tracks : list of strings
          List containing the songs under the selected directory
        """

        if self.metadata is None:
            # Make sure metadata has been loaded
            self.load_metadata()

        # Initialize a list to hold valid tracks
        tracks = list()

        for track in self.metadata.keys():
            # Check if track has the specified genre
            if split in self.metadata[track]:
                # Add the track name
                tracks.append(track)

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          FMA track name

        Returns
        ----------
        mp3_path : string
          Path to audio for the specified track
        """

        # Construct the path to the MP3 file
        mp3_path = os.path.join(self.base_dir, track[:3], f'{track}.mp3')

        return mp3_path

    @classmethod
    def name(cls):
        """
        Simple helper function to get base class name.

        Returns
        ----------
        name : string
          Identifier for the dataset
        """

        # Use name of parent class
        tag = cls.__base__.__name__

        return tag

    @classmethod
    def download(cls, save_dir):
        """
        Download the FMA dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of FMA
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file containing all the metadata
        meta_url = f'https://os.unil.cloud.switch.ch/fma/fma_metadata.zip'

        # Construct a path for saving the metadata
        meta_path = os.path.join(save_dir, os.path.basename(meta_url))

        # Download the metadata zip file
        stream_url_resource(meta_url, meta_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(meta_path)

        # URL pointing to the zip file containing excerpts for all tracks
        audio_url = f'https://os.unil.cloud.switch.ch/fma/fma_{cls.SIZE}.zip'

        # Construct a path for saving the audio
        audio_path = os.path.join(save_dir, os.path.basename(audio_url))

        # Download the audio zip file
        stream_url_resource(audio_url, audio_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(audio_path)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, f'fma_{cls.SIZE}'))


class FMA_F(FMA):
    """
    Wrapper to download full dataset.
    """

    SIZE = 'full'


class FMA_L(FMA):
    """
    Wrapper to download large version.
    """

    SIZE = 'large'


class FMA_M(FMA):
    """
    Wrapper to download medium version.
    """

    SIZE = 'medium'


class FMA_S(FMA):
    """
    Wrapper to download small version.
    """

    SIZE = 'small'
