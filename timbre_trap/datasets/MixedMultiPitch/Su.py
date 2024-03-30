from .. import AMTDataset

import numpy as np
import librosa
import os


class Su(AMTDataset):
    """
    Implements a wrapper for the Su dataset.
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Prefix indicators for each piece in the dataset
        """

        splits = ['PQ02', 'PQ03', 'PS01', 'PS02', 'PS03',
                  'SQ01', 'SQ02', 'SQ03', 'VS01', 'VS04']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Individual track prefix

        Returns
        ----------
        tracks : list of strings
          List containing the individual track specified
        """

        # Construct the path to the directory containing audio
        audio_dir = os.path.join(self.base_dir, 'audio')

        # Obtain the full name corresponding to the track prefix
        tracks = [d.replace('_audio.wav', '') for d in os.listdir(audio_dir) if d.startswith(split)]

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          Su track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Construct the path to the audio mixture
        wav_path = os.path.join(self.base_dir, 'audio', f'{track}_audio.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          Su track name

        Returns
        ----------
        txt_path : string
          Path to ground-truth for the specified track
        """

        # Construct the path to the note annotations for the track
        txt_path = os.path.join(self.base_dir, 'gt_Note', f'{track}_note.txt')

        return txt_path

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          Su track name

        Returns
        ----------
        pitches : ndarray (L)
          Array of note pitches
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        """

        # Obtain the path to the track's ground_truth
        txt_path = self.get_ground_truth_path(track)

        # Open annotations in reading mode
        with open(txt_path) as txt_file:
            # Read note-level annotations into a list
            notes = [f.split() for f in txt_file.readlines()]

        # Extract the MIDI pitch for each note entry
        pitches = librosa.hz_to_midi(np.array([n.pop(-1) for n in notes]).astype('float'))

        # Pack the remaining intervals into an array
        intervals = np.array(notes).astype('float')

        return pitches, intervals

    @classmethod
    def download(cls, save_dir):
        """
        At this time, Su must be downloaded manually so an error is thrown.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of Su
        """

        return NotImplementedError
