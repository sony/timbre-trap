from timbre_trap.utils.data import *
from .. import AMTDataset

import pandas as pd
import numpy as np
import os


class MusicNet(AMTDataset):
    """
    Implements the top-level wrapper for the MusicNet dataset (https://zenodo.org/record/5120004).
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Names of originally proposed splits
        """

        splits = ['train', 'test']

        return splits

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
          Names of tracks belonging to the split
        """

        # Obtain tracks names as files under the split's data directory
        tracks = os.listdir(os.path.join(self.base_dir, f'{split}_data'))

        # Remove the file extension and add the split to all files
        tracks = sorted([os.path.join(split, os.path.splitext(t)[0]) for t in tracks])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MusicNet track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Break apart partition and track name
        split, name = os.path.split(track)

        # Get the path to the audio
        wav_path = os.path.join(self.base_dir, f'{split}_data', f'{name}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground truth.

        Parameters
        ----------
        track : string
          MusicNet track name

        Returns
        ----------
        csv_path : string
          Path to ground-truth for the specified track
        """

        # Break apart partition and track name
        split, name = os.path.split(track)

        # Get the path to the ground-truth note annotations
        csv_path = os.path.join(self.base_dir, f'{split}_labels', f'{name}.csv')

        return csv_path

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          MusicNet track name

        Returns
        ----------
        pitches : ndarray (L)
          Array of note pitches
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        """

        # Obtain the path to the track's ground_truth
        csv_path = self.get_ground_truth_path(track)

        # Load tabulated note data from the csv file
        note_entries = pd.read_csv(csv_path).to_numpy()

        # Unpack the relevant note attributes and convert them to integers
        onsets, offsets, pitches = note_entries[:, (0, 1, 3)].T.astype(int)

        # Construct intervals for the notes and convert to seconds
        intervals = np.concatenate(([onsets], [offsets])).T / 44100

        return pitches, intervals

    @classmethod
    def download(cls, save_dir):
        """
        Download the MusicNet dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MusicNet
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the tar file containing audio/annotations
        anno_url = 'https://zenodo.org/record/5120004/files/musicnet.tar.gz'

        # Construct a path for saving the annotations
        anno_path = os.path.join(save_dir, os.path.basename(anno_url))

        # Download the tar file containing annotations
        stream_url_resource(anno_url, anno_path, 1000 * 1024)

        # Untar the downloaded file and remove it
        unzip_and_remove(anno_path, tar=True)

        # Move contents of untarred directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'musicnet'))

        # URL pointing to the tar file containing MIDI for all tracks
        midi_url = 'https://zenodo.org/record/5120004/files/musicnet_midis.tar.gz'

        # Construct a path for saving the MIDI files
        midi_path = os.path.join(save_dir, os.path.basename(midi_url))

        # Download the tar file containing MIDI files
        stream_url_resource(midi_url, midi_path, 1000 * 1024)

        # Untar the downloaded file and remove it
        unzip_and_remove(midi_path, tar=True)

        # URL pointing to the metadata file for all tracks
        meta_url = 'https://zenodo.org/record/5120004/files/musicnet_metadata.csv'

        # Construct a path for saving the metadata
        meta_path = os.path.join(save_dir, os.path.basename(meta_url))

        # Download the metadata file
        stream_url_resource(meta_url, meta_path, 1000 * 1024)
