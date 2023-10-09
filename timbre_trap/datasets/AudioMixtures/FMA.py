from .. import AudioDataset
from ..utils import *

import numpy as np
import os


class FMA(AudioDataset):
    """
    Implements a wrapper for the Free Music Archive dataset (https://github.com/mdeff/fma).
    """

    SIZE = None

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Top-level directories from download
        """

        # All numbered directories with leading zeros
        splits = [str(i).zfill(3) for i in np.arange(156)]

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

        # Construct a path to the dataset split
        split_path = os.path.join(self.base_dir, split)

        # Obtain a sorted list of all files in the split's directory
        tracks = sorted([os.path.splitext(f)[0] for f in os.listdir(split_path)])

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

        # Get the path to the MP3 file
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

        # URL pointing to the zip file containing excerpts for all tracks
        url = f'https://os.unil.cloud.switch.ch/fma/fma_{cls.SIZE}.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)

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
