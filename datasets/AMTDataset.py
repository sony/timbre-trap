from . import AudioDataset, NoteDataset, constants

from abc import abstractmethod

import numpy as np
import librosa
import torch


class AMTDataset(AudioDataset, NoteDataset):
    """
    Implements functionality for a dataset with audio and note annotations.
    """

    @abstractmethod
    def __getitem__(self, index):
        """
        Extract the audio and ground-truth data for a sampled track.

        Parameters
        ----------
        index : int
          Index of sampled track

        Returns
        ----------
        data : dict containing
          track : string
            Identifier for the track
          audio : Tensor (1 x N)
            Audio data read for the track
          times : Tensor (T)
            Time associated with each frame
          ground_truth : Tensor (F x T)
            Ground-truth activations for the track
        """

        # Determine corresponding track
        track = self.tracks[index]

        try:
            # Attempt to read the track's audio
            audio = self.get_audio(track)
        except Exception as e:
            # Print offending track to console
            print(f'Error loading track \'{track}\': {repr(e)}')

            # Default audio to silence
            audio = torch.empty((1, 0))

        # Read the track's note annotations
        pitches, intervals = self.get_ground_truth(track)

        # Convert note pitches to Hertz
        pitches = librosa.midi_to_hz(pitches)

        # Determine frame times given the number of frames within the audio
        times = self.cqt.get_times(self.cqt.get_expected_frames(audio.shape[-1]))

        if self.n_secs is not None:
            # Determine the required sequence length
            n_samples = self.cqt.get_expected_samples(self.n_secs)
            # Determine the corresponding number of frames
            n_frames = self.cqt.get_expected_frames(n_samples)

            if audio.size(-1) >= n_samples:
                # Sample a random starting index for the trim
                start = self.rng.randint(0, audio.size(-1) - n_samples + 1)
                # Trim audio to the sequence length
                audio = audio[..., start : start + n_samples]
                # Determine the time of the sample
                offset_time = start / self.sample_rate
                # Compute the associated frame times
                times = self.cqt.get_times(n_frames) + offset_time
            else:
                # Determine how much padding is required
                pad_total_s = n_samples - audio.size(-1)
                # Randomly distribute between both sides
                pad_left_s = self.rng.randint(0, pad_total_s)
                # Pad the audio with zeros
                audio = torch.nn.functional.pad(audio, (pad_left_s, pad_total_s - pad_left_s))

                # Determine the time shift caused by padding a fraction of a full frame
                offset_time = (pad_left_s % self.cqt.hop_length) / self.sample_rate
                # Correct the times
                times -= offset_time

        # Convert note annotations to multi pitch annotations
        multi_pitch = self.notes_to_multi_pitch(pitches, intervals, times)

        # Convert pitch list observations to multi pitch activations
        ground_truth = self.multi_pitch_to_activations(multi_pitch, self.cqt.midi_freqs)

        if self.n_secs is not None and len(times) != n_frames:
            # Determine number of frames made entirely of padding
            pad_left_f = int(pad_left_s // self.cqt.hop_length)
            # Compute remaining number of frames to be padded
            pad_right_f = n_frames - len(times) - pad_left_f

            # Pad the times with -1 to indicate invalid times
            times = np.pad(times, (pad_left_f, pad_right_f), constant_values=-1)
            # Pad the final dimension of the ground-truth with zeros
            ground_truth = np.pad(ground_truth, ((0, 0), (pad_left_f, pad_right_f)))

        # Pack the data into a dictionary
        data = {constants.KEY_TRACK : track,
                constants.KEY_AUDIO : audio,
                constants.KEY_TIMES : times,
                constants.KEY_GROUND_TRUTH : ground_truth}

        return data
