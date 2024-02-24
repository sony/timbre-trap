from timbre_trap.utils.data import *
from .. import AMTDataset

import pandas as pd
import numpy as np
import mido
import os


class MAESTRO(AMTDataset):
    """
    Implements a wrapper for the MAESTRO dataset (V3)
    (https://magenta.tensorflow.org/datasets/maestro).
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

        splits = ['train', 'validation', 'test']

        return splits

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          String indicating train, validation, or test split

        Returns
        ----------
        tracks : list of strings
          Names of tracks belonging to the split
        """

        # Load tabulated metadata from the csv file under the top-level directory
        csv_data = pd.read_csv(os.path.join(self.base_dir, 'maestro-v3.0.0.csv'))

        # Obtain a list of the track names and their corresponding splits
        names, associations = csv_data['audio_filename'], csv_data['split']
        # Filter out tracks that do not belong to the specified data split
        tracks = [t for t, a in zip(names, associations) if a == split]
        # Remove the file extensions from each track and sort the list
        tracks = sorted([os.path.splitext(track)[0] for track in tracks])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MAESTRO track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Get the path to the audio recording
        wav_path = os.path.join(self.base_dir, f'{track}.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          MAESTRO track name

        Returns
        ----------
        midi_path : string
          Path to ground-truth for the specified track
        """

        # Get the path to the note annotations for the performance
        midi_path = os.path.join(self.base_dir, f'{track}.midi')

        return midi_path

    @staticmethod
    def load_notes_midi(midi_path):
        """
        Load all notes from a MIDI file and account for sustain pedal activity.

        Parameters
        ----------
        midi_path : string
          Path to MIDI file to read

        Returns
        ----------
        pitches : ndarray (L)
          Array of note pitches
        velocities : ndarray (L)
          Array of corresponding onset velocities
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        """

        # Open the MIDI file and read all messages
        midi = mido.MidiFile(midi_path)

        # Initialize a counter for the time
        time = 0

        # Keep track of sustain pedal activity
        sustain_status = False

        # Initialize an empty list to store events
        events = []

        # Loop through MIDI messages
        for message in midi:
            # Increment the time accordingly
            time += message.time

            # Check if the message is a control change event
            if message.type == 'control_change':
                # Determine if it is a sustain control event
                sustain_control = message.control == 64

                # Check if the message corresponds to SUSTAIN_ON or SUSTAIN_OFF
                # (value >= 64 => SUSTAIN_ON | value < 64 => SUSTAIN OFF)
                sustain_on = message.value >= 64

                # Check if sustain pedal status has actually changed
                sustain_change = sustain_on != sustain_status

                if sustain_control and sustain_change:
                    # Update sustain pedal status
                    sustain_status = sustain_on
                    # Determine which event occurred (SUSTAIN_ON or SUSTAIN_OFF)
                    event_type = 'sustain_on' if sustain_status else 'sustain_off'

                    # Create a new event detailing the sustain pedal activity
                    event = dict(index=len(events),
                                 time=time,
                                 type=event_type,
                                 note=None,
                                 velocity=0)
                    # Add the sustain pedal event to the MIDI event list
                    events.append(event)

            # Check if the message is a note event
            if 'note' in message.type:
                # Set the velocity according to the message type
                # (offsets are NOTE_OFF events or NOTE_ON with velocity = 0)
                velocity = message.velocity if message.type == 'note_on' else 0

                # Create a new event detailing the note activity
                event = dict(index=len(events),
                             time=time,
                             type='note',
                             note=message.note,
                             velocity=velocity,
                             sustain=sustain_status)
                # Add the note event to the MIDI event list
                events.append(event)

        # Initialize empty arrays to store attributes for each note
        pitches, velocities, intervals = np.empty(0), np.empty(0), np.empty((0, 2))

        # Loop through all documented events
        for i, onset in enumerate(events):
            # Build notes starting at onsets
            if onset['velocity'] == 0:
                continue

            # Set the corresponding offset as the next note event with the same pitch or the final event
            offset = next(n for n in events[i + 1:] if n['note'] == onset['note'] or n is events[-1])

            # Check if sustain pedal was active during the offset
            if offset['sustain'] and offset is not events[-1]:
                # Offset occurs when sustain ends or another note event of same pitch occurs
                offset = next(n for n in events[offset['index'] + 1:] if n['type'] == 'sustain_off' or
                              n['note'] == onset['note'] or n is events[-1])

            # Append the note attributes to the respective arrays
            pitches = np.append(pitches, onset['note'])
            velocities = np.append(velocities, onset['velocity'])
            intervals = np.concatenate((intervals, [[onset['time'], offset['time']]]))

        return pitches, velocities, intervals

    def get_ground_truth(self, track):
        """
        Extract the ground-truth for the specified track.

        Parameters
        ----------
        track : string
          MAESTRO track name

        Returns
        ----------
        pitches : ndarray (L)
          Array of note pitches
        intervals : ndarray (L x 2)
          Array of corresponding onset-offset time pairs
        """

        # Obtain the path to the track's ground_truth
        midi_path = self.get_ground_truth_path(track)

        # Parse the notes from the MIDI annotations
        pitches, _, intervals = self.load_notes_midi(midi_path)

        return pitches, intervals

    @classmethod
    def download(cls, save_dir):
        """
        Download the MAESTRO dataset (V3) to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MAESTRO
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file for version 3 of the MAESTRO dataset
        url = f'https://storage.googleapis.com/magentadata/datasets/maestro/v3.0.0/maestro-v3.0.0.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'maestro-v3.0.0'))
