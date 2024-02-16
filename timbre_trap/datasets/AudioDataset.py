from . import BaseDataset, constants

from abc import abstractmethod

import torchaudio
import torch


class AudioDataset(BaseDataset):
    """
    Implements functionality for a dataset with audio.
    """

    def __init__(self, sample_rate=16000, **kwargs):
        """
        Instantiate the dataset wrapper.

        Parameters
        ----------
        See BaseDataset for others...

        sample_rate : int
          Desired sample rate for the audio
        """

        BaseDataset.__init__(self, **kwargs)

        self.sample_rate = sample_rate

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

        try:
            # Load the audio with torchaudio
            audio, fs = torchaudio.load(audio_path)
            # Average channels to obtain mono-channel
            audio = torch.mean(audio, dim=0, keepdim=True)
            # Resample audio to the specified sampling rate
            audio = torchaudio.functional.resample(audio, fs, self.sample_rate)

            if audio.abs().max():
                # Normalize the audio using the infinity norm
                audio /= audio.abs().max()

        except Exception as e:
            # Print offending track to console
            print(f'Error loading track \'{track}\': {repr(e)}')

            # Default audio to silence
            audio = torch.empty((1, 0))

        return audio

    def slice_audio(self, audio, n_samples=None, offset_s=None):
        """
        Slice a piece of audio to a specified size.

        Parameters
        ----------
        audio : Tensor (1 x N)
          Audio data to slice
        n_samples : int or None (Optional)
          Number of samples to slice
        offset_s : int or None (Optional)
          Offset (in samples) for slice

        Returns
        ----------
        audio : Tensor (1 x M)
          Audio data sliced accordingly
        offset_t : float
          Offset (in seconds) used to take slice
        """

        if n_samples is None:
            # Default slice to preselected sequence length
            n_samples = int(self.n_secs * self.sample_rate)

        if audio.size(-1) >= n_samples:
            if offset_s is None:
                # Sample a starting sample index randomly for the trim
                start = self.rng.randint(0, audio.size(-1) - n_samples + 1)
            else:
                # Use provided value
                start = offset_s

            # Determine corresponding time offset
            offset_t = start / self.sample_rate

            # Trim audio to the sequence length
            audio = audio[..., start : start + n_samples]
        else:
            # Determine how much padding is required
            pad_total = n_samples - audio.size(-1)

            if offset_s is None:
                # Randomly distribute padding
                pad_left = self.rng.randint(0, pad_total)
            else:
                # Use provided value
                pad_left = abs(offset_s)

            # Determine corresponding time offset
            offset_t = -pad_left / self.sample_rate

            # Pad the audio with zeros on both sides by sampled amount
            audio = torch.nn.functional.pad(audio, (pad_left, pad_total - pad_left))

        return audio, offset_t

    @abstractmethod
    def __getitem__(self, index, offset_s=None):
        """
        Extract the audio data for a sampled track.

        Parameters
        ----------
        index : int
          Index of sampled track
        offset_s : int or None (Optional)
          Offset (in samples) for slice

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

        # Load the track's audio
        audio = self.get_audio(track)

        if self.n_secs is not None:
            # Randomly slice audio using default sequence length
            audio, _ = self.slice_audio(audio, offset_s=offset_s)

        # Pack the data into a dictionary
        data = {constants.KEY_TRACK : track,
                constants.KEY_AUDIO : audio}

        return data
