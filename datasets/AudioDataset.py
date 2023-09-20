from . import BaseDataset, constants

from abc import abstractmethod

import torchaudio
import torch


class AudioDataset(BaseDataset):
    """
    Implements functionality for a dataset with audio.
    """

    @abstractmethod
    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          Track name

        Returns
        ----------
        audio_path : string
          Path to audio for the specified track
        """

        return NotImplementedError

    def get_audio(self, track):
        """
        Extract the audio for the specified track.

        Parameters
        ----------
        track : string
          Track name

        Returns
        ----------
        audio : Tensor (1 x N)
          Audio data read for the track
        """

        # Obtain the path to the track's audio
        audio_path = self.get_audio_path(track)

        # Load the audio with torchaudio
        audio, fs = torchaudio.load(audio_path)
        # Average channels to obtain mono-channel
        audio = torch.mean(audio, dim=0, keepdim=True)
        # Resample audio to the specified sampling rate
        audio = torchaudio.functional.resample(audio, fs, self.sample_rate)

        if audio.abs().max():
            # Normalize the audio using the infinity norm
            audio /= audio.abs().max()

        return audio

    @abstractmethod
    def __getitem__(self, index):
        """
        Extract the audio data for a sampled track.

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

        if self.n_secs is not None:
            # Determine the required sequence length
            n_samples = int(self.n_secs * self.sample_rate)

            if audio.size(-1) >= n_samples:
                # Sample a random starting index for the trim
                start = self.rng.randint(0, audio.size(-1) - n_samples + 1)
                # Trim audio to the sequence length
                audio = audio[..., start : start + n_samples]
            else:
                # Determine how much padding is required
                pad_total = n_samples - audio.size(-1)
                # Randomly distribute between both sides
                pad_left = self.rng.randint(0, pad_total)
                # Pad the audio with zeros
                audio = torch.nn.functional.pad(audio, (pad_left, pad_total - pad_left))

        # Pack the data into a dictionary
        data = {constants.KEY_TRACK : track,
                constants.KEY_AUDIO : audio}

        return data
