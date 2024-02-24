from timbre_trap.utils.data import *
from .. import MPEDataset

import numpy as np
import jams
import os


class GuitarSet(MPEDataset):
    """
    Implements a wrapper for the GuitarSet dataset (https://guitarset.weebly.com/).
    """

    SAMPLING_RATE = 44100
    HOP_LENGTH = 256

    def __init__(self, **kwargs):
        """
        Override the resampling indices to prevent erroneous extension of pitches.
        """

        # Determine if resampling indices were provided
        resample_idcs = kwargs.pop('resample_idcs', None)

        if resample_idcs is None:
            # Change the default resampling indices
            resample_idcs = [0, 0]

        # Update the value within the keyword arguments
        kwargs.update({'resample_idcs' : resample_idcs})

        super().__init__(**kwargs)

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          IDs of the six different guitarists
        """

        splits = ['00', '01', '02', '03', '04', '05']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          String indicating guitarist ID

        Returns
        ----------
        tracks : list of strings
          Names of tracks belonging to the split
        """

        # Extract the names of all the files in annotation directory
        jams_files = os.listdir(os.path.join(self.base_dir, 'annotation'))

        # Filter out files with a mismatching player ID and remove JAMS extension
        tracks = [os.path.splitext(t)[0] for t in jams_files if t.startswith(split)]

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          GuitarSet track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Get the path to the audio recording
        wav_path = os.path.join(self.base_dir, 'audio_mono-mic', f'{track}_mic.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          GuitarSet track name

        Returns
        ----------
        jams_path : string
          Path to ground-truth for the specified track
        """

        # Get the path to the F0 annotations for the performance
        jams_path = os.path.join(self.base_dir, 'annotation', f'{track}.jams')

        return jams_path

    @staticmethod
    def load_pitches_jams(jams_path):
        """
        Load all pitch observations from a JAMS file and align them with a uniform time grid.

        Parameters
        ----------
        jams_path : string
          Path to JAMS file to read

        Returns
        ----------
        times : ndarray (T)
          Time associated with each observation
        pitches : list of ndarray (T x [...])
          Multi-pitch annotations in Hertz
        """

        # Load all the JAMS data
        jam = jams.load(jams_path)

        # Extract the pitch annotations by string
        pitch_data_slices = jam.annotations['pitch_contour']

        # Determine track's total duration
        duration = jam.file_metadata.duration

        # Compute time interval between annotations in seconds
        hop_length_s = GuitarSet.HOP_LENGTH / GuitarSet.SAMPLING_RATE

        # Determine the total number of frames for a uniform time series
        num_entries = int(np.ceil(duration / hop_length_s)) + 1

        # Initialize array of times corresponding to uniform frames
        times = hop_length_s * np.arange(num_entries)

        # Initialize empty list of pitch observations
        pitches = [np.array([])] * num_entries

        # Loop through string-level annotations
        for i in range(len(pitch_data_slices)):
            # Extract the pitch observations
            annotations = pitch_data_slices[i]

            # Loop through pitch observations
            for pitch in annotations:
                # Represent pitch within an array
                freq = np.array([pitch.value['frequency']])

                # Ignore zero or unvoiced frequencies
                if np.sum(freq) == 0 or not pitch.value['voiced']:
                    # Replace with empty array
                    freq = np.empty(0)

                # Determine closest index for the annotation
                closest_idx = np.argmin(np.abs(times - pitch.time))

                # Append pitch observation to the appropriate frame
                pitches[closest_idx] = np.append(pitches[closest_idx], freq)

        return times, pitches

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          GuitarSet track name

        Returns
        ----------
        times : ndarray (T)
          Time associated with each frame of annotations
        pitches : list of ndarray (T x [...])
          Frame-level multi-pitch annotations in Hertz
        """

        # Obtain the path to the track's ground_truth
        jams_path = self.get_ground_truth_path(track)

        # Parse the pitch activity from the JAMS annotations
        times, pitches = self.load_pitches_jams(jams_path)

        return times, pitches

    @classmethod
    def download(cls, save_dir):
        """
        Download the GuitarSet dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of GuitarSet
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file containing annotations for all tracks
        anno_url = 'https://zenodo.org/record/3371780/files/annotation.zip'

        # Construct a path to directory to hold annotations
        anno_dir = os.path.join(save_dir, 'annotation')
        # Create the annotation directory
        os.makedirs(anno_dir)

        # Construct a path for saving the annotations
        anno_path = os.path.join(anno_dir, os.path.basename(anno_url))

        # Download the zip file
        stream_url_resource(anno_url, anno_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(anno_path)

        # URL pointing to the zip file containing audio for all tracks
        audio_url = 'https://zenodo.org/record/3371780/files/audio_mono-mic.zip'

        # Construct a path to directory to hold audio
        audio_dir = os.path.join(save_dir, 'audio_mono-mic')
        # Create the annotation directory
        os.makedirs(audio_dir)

        # Construct a path for saving the audio
        audio_path = os.path.join(audio_dir, os.path.basename(audio_url))

        # Download the zip file
        stream_url_resource(audio_url, audio_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(audio_path)
