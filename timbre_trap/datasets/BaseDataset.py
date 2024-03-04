from timbre_trap.utils.data import *

from torch.utils.data import Dataset
from abc import abstractmethod

import numpy as np
import warnings
import shutil
import torch
import os


class BaseDataset(Dataset):
    """
    Implements bare minimum functionality for a usable dataset.
    """

    def __init__(self, base_dir=None, splits=None, n_secs=None, seed=0):
        """
        Instantiate the dataset wrapper.

        Parameters
        ----------
        base_dir : string
          Path to the top-level directory
        splits : list of strings
          Dataset partitions to include
        n_secs : float
          Sequence length for sampling
        seed : int
          Seed for random sampling
        """

        if base_dir is None:
            # Assume the dataset exists at the default location
            base_dir = os.path.join(constants.DEFAULT_LOCATION, self.name())

        self.base_dir = base_dir

        # Check if the dataset exists at specified path
        if not os.path.isdir(self.base_dir):
            warnings.warn(f'Could not find dataset at specified path \'{self.base_dir}\''
                          '. Attempting to download...', category=RuntimeWarning)
            # Attempt to download the dataset
            self.download(self.base_dir)

        if splits is None:
            # Use all available dataset splits
            splits = self.available_splits()

        self.tracks = []

        for split in splits:
            # Aggregate track names from selected splits
            self.tracks += self.get_tracks(split)

        self.n_secs = n_secs

        # Initialize random number generator
        self.rng = np.random.RandomState(seed)

    @classmethod
    @abstractmethod
    def name(cls):
        """
        Simple helper function to get the class name.

        Returns
        ----------
        name : string
          Identifier for the dataset
        """

        name = cls.__name__

        return name

    @staticmethod
    @abstractmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Partitions of dataset
        """

        return NotImplementedError

    @abstractmethod
    def get_tracks(self, split):
        """
        Get the track names belonging to a dataset partition.

        Parameters
        ----------
        split : string
          Individual dataset partition

        Returns
        ----------
        tracks : list of strings
          Names of tracks belonging to the partition
        """

        return NotImplementedError

    def __len__(self):
        """
        Defines the number of individual tracks within the dataset that can be sampled.

        Returns
        ----------
        length : int
          Number of tracks
        """

        # Count the track names
        length = len(self.tracks)

        return length

    @abstractmethod
    def __getitem__(self, index):
        """
        Extract the data for a sampled track.

        Parameters
        ----------
        index : int
          Index of sampled track
        """

        return NotImplementedError

    @classmethod
    @abstractmethod
    def download(cls, save_dir):
        """
        Helper to create the top-level directory for inherited datasets.

        Parameters
        ----------
        save_dir : string
          Directory under which to save dataset contents
        """

        if os.path.isdir(save_dir):
            # Remove directory if it already exists
            shutil.rmtree(save_dir)

        # Create the base directory
        os.makedirs(save_dir)

        print(f'Downloading {cls.__name__}...')


class ComboDataset(Dataset):
    """
    Wrapper to train with multiple pre-instantiated datasets.
    """

    def __init__(self, datasets):
        """
        Instantiate the combination wrapper.

        Parameters
        ----------
        datasets : list of BaseDataset
          Pre-instantiated datasets from which to sample
        """

        self.datasets = datasets

    def __len__(self):
        """
        Number of samples across all datasets.

        Returns
        ----------
        length : int
          Total number of tracks
        """

        # Add together length of combined datasets
        length = sum([len(d) for d in self.datasets])

        return length

    def __getitem__(self, index):
        """
        Extract the data for a sampled track.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        data : dict containing
          track : string
            Identifier for the track
          ...
        """

        # Keep track of relative index
        local_idx, dataset_idx = index, 0

        while local_idx >= self.datasets[dataset_idx].__len__():
            # Subtract the length of the dataset from global index
            local_idx -= self.datasets[dataset_idx].__len__()
            # Check next dataset
            dataset_idx += 1

        # Sample data at the local index of selected dataset
        data = self.datasets[dataset_idx].__getitem__(local_idx)

        return data


class StemMixingDataset(ComboDataset):
    """
    Wrapper to train on randomly mixed pre-instantiated datasets.
    """

    def __init__(self, datasets, tracks_per_epoch, n_min=2, n_max=5, seed=0):
        """
        Instantiate the stem mixing wrapper.

        Parameters
        ----------
        datasets : list of BaseDataset
          Pre-instantiated datasets from which to sample
        tracks_per_epoch : int
          Number of mixtures to create within a single epoch
        n_min : int
          Minimum number of stems to include in a random mix
        n_max : int
          Maximum number of stems to include in a random mix
        seed : int
          Seed for random mixing
        """

        super().__init__(datasets)

        # Verify all datasets have same sequence length set
        assert len(set([d.n_secs for d in self.datasets])) == 1

        self.tracks_per_epoch = tracks_per_epoch
        self.n_min = n_min
        self.n_max = n_max

        # Initialize random number generator for mixing
        self.rng = np.random.RandomState(seed)

    def __len__(self):
        """
        Number of mixtures to make per epoch.

        Returns
        ----------
        length : int
          Total number of tracks
        """

        # Length was set in constructor
        length = self.tracks_per_epoch

        return length

    def __getitem__(self, index):
        """
        Extract the data for a sampled track.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        data : dict containing
          track : string
            Identifier for the track
          ...
        """

        # Determine how many stems to mix for this sample
        n_mix = self.rng.randint(self.n_min, self.n_max + 1)

        # Obtain list of indices for all stems
        track_idcs = np.arange(super().__len__())
        # Shuffle the order of the stems
        self.rng.shuffle(track_idcs)

        # Extract data from selected amount of random stems
        stems = [ComboDataset.__getitem__(self, i) for i in track_idcs[:n_mix]]

        # Separate track data by availability of MPE ground-truth
        data_both, data_audio, _ = separate_ground_truth(stems)

        # Initialize a dictionary for mixture
        data = {constants.KEY_TRACK : str(index),
                constants.KEY_AUDIO : None}

        if data_audio is not None:
            # Sum batched stem audio together to obtain a mixture
            data[constants.KEY_AUDIO] = torch.sum(data_audio[constants.KEY_AUDIO], dim=0)

        if data_both is not None:
            # Sum batched stem audio together to obtain a mixture
            mpe_audio = torch.sum(data_both[constants.KEY_AUDIO], dim=0)

            if data[constants.KEY_AUDIO] is None:
                # Insert the audio mixture into the dictionary
                data[constants.KEY_AUDIO] = mpe_audio
            else:
                # Add the audio to the prexisting mixture
                data[constants.KEY_AUDIO] += mpe_audio

            # Combine all multi-pitch annotations into a single representation and clamp activations
            data[constants.KEY_GROUND_TRUTH] = torch.sum(data_both[constants.KEY_GROUND_TRUTH], dim=0).clamp(0, 1)

            # Add placeholder frame times to mixture data
            data[constants.KEY_TIMES] = data_both[constants.KEY_TIMES][0]

            # Convert data to NumPy arrays for consistency
            data[constants.KEY_TIMES] = np.array(data[constants.KEY_TIMES])
            data[constants.KEY_GROUND_TRUTH] = np.array(data[constants.KEY_GROUND_TRUTH])

        return data
