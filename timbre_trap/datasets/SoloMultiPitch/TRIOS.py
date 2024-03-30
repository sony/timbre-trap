from ..Common import TRIOS
from ..SoloMultiPitch import MAESTRO

import os


class TRIOS(TRIOS):
    """
    Implements a wrapper for the TRIOS dataset stems.
    """

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Multitrack name

        Returns
        ----------
        tracks : list of strings
          List containing the instruments of the multitrack
        """

        # Obtain a list of all files under the multitrack directory
        all_files = os.listdir(os.path.join(self.base_dir, split))

        # Obtain a list of all instruments within the multitrack
        instruments = [os.path.splitext(i)[0] for i in all_files if i.endswith('.mid')]
        # Discard any unpitched instruments when constructing list of tracks
        tracks = [os.path.join(split, i) for i in instruments if i in self.PITCHED_INSTRUMENTS]

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          TRIOS track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Split the track name to obtain hierarchy
        multitrack, instrument = os.path.split(track)

        # Construct the path to the audio of the stem
        wav_path = os.path.join(self.base_dir, multitrack, f'{instrument}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          TRIOS track name

        Returns
        ----------
        midi_path : string
          Path to ground-truth for the specified track
        """

        # Split the track name to obtain hierarchy
        multitrack, instrument = os.path.split(track)

        # Construct the path to the note annotations for the stem
        midi_path = os.path.join(self.base_dir, multitrack, f'{instrument}.mid')

        return midi_path

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          TRIOS track name

        Returns
        ----------
        pitches : ndarray (L)
          Array of note pitches
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        """

        # Obtain the path to the ground_truth
        midi_path = self.get_ground_truth_path(track)

        # Parse the notes from the MIDI annotations
        pitches, _, intervals = MAESTRO.load_notes_midi(midi_path)

        return pitches, intervals
