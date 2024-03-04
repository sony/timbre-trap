from ..Common import TRIOS
from ..SoloMultiPitch import MAESTRO

import numpy as np
import os


class TRIOS(TRIOS):
    """
    Implements a wrapper for the TRIOS dataset multitracks.
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
          List containing the multitrack name
        """

        # Split is track name
        tracks = [split]

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

        # Construct the path to the multitrack audio
        wav_path = os.path.join(self.base_dir, track, 'mix.wav')

        return wav_path

    def get_ground_truth_path(self, track, instrument):
        """
        Get the path to a track's ground-truth for a specific instrument.

        Parameters
        ----------
        track : string
          TRIOS track name
        instrument : string
          Instrument within track

        Returns
        ----------
        midi_path : string
          Path to ground-truth for the specified track and instrument
        """

        # Construct the path to the note annotations for the instrument
        midi_path = os.path.join(self.base_dir, track, f'{instrument}.mid')

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

        # Obtain a list of all files under the multitrack directory
        all_files = os.listdir(os.path.join(self.base_dir, track))

        # Obtain a list of all instruments within the multitrack
        instruments = [os.path.splitext(i)[0] for i in all_files if i.endswith('.mid')]
        # Discard any unpitched instruments when constructing ground-truth
        valid_instruments = [i for i in instruments if i in self.PITCHED_INSTRUMENTS]

        # Initialize arrays to hold note attributes
        pitches, intervals = np.empty(0), np.empty((0, 2))

        # Loop through pitched instruments
        for instrument in valid_instruments:
            # Obtain the path to the ground_truth for the instrument
            midi_path = self.get_ground_truth_path(track, instrument)

            # Parse the notes from the MIDI annotations
            pitches_, _, intervals_ = MAESTRO.load_notes_midi(midi_path)

            # Add instrument's notes to collection
            pitches = np.append(pitches, pitches_)
            intervals = np.append(intervals, intervals_, axis=0)

        return pitches, intervals
