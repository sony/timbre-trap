from torchaudio.transforms import AmplitudeToDB
from cqt_pytorch import CQT as _CQT

import numpy as np
import librosa
import torch
import math


class CQT(_CQT):
    """
    Wrapper which adds some basic functionality to the sliCQ module.
    """

    def __init__(self, n_octaves, bins_per_octave, sample_rate, secs_per_block):
        """
        Instantiate the sliCQ module and wrapper.

        Parameters
        ----------
        n_octaves : int
          Number of octaves below Nyquist to span
        bins_per_octave : int
          Number of bins allocated to each octave
        sample_rate : int or float
          Number of samples per second of audio
        secs_per_block : float
          Number of seconds to process at a time
        """

        super().__init__(num_octaves=n_octaves,
                         num_bins_per_octave=bins_per_octave,
                         sample_rate=sample_rate,
                         block_length=int(secs_per_block * sample_rate),
                         power_of_2_length=True)

        self.sample_rate = sample_rate

        # Compute hop length corresponding to transform coefficients
        self.hop_length = (self.block_length / self.max_window_length)

        # Compute total number of bins
        self.n_bins = n_octaves * bins_per_octave
        # Determine frequency (MIDI) below Nyquist by specified octaves
        fmin = librosa.hz_to_midi((sample_rate / 2) / (2 ** n_octaves))

        # Determine center frequency (MIDI) associated with each bin of module
        self.midi_freqs = fmin + np.arange(self.n_bins) / (bins_per_octave / 12)

    def forward(self, audio):
        """
        Encode a batch of audio into CQT spectral coefficients.

        Parameters
        ----------
        audio : Tensor (B x 1 X T)
          Batch of input audio

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of real/imaginary CQT coefficients
        """

        with torch.no_grad():
            # Obtain complex CQT coefficients
            coefficients = self.encode(audio)

            # Convert complex coefficients to real representation
            coefficients = self.to_real(coefficients)

        return coefficients

    @staticmethod
    def to_real(coefficients):
        """
        Convert a set of complex coefficients to equivalent real representation.

        Parameters
        ----------
        coefficients : Tensor (B x 1 x F X T)
          Batch of complex CQT coefficients

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of real/imaginary CQT coefficients
        """

        # Collapse channel dimension (mono assumed)
        coefficients = coefficients.squeeze(-3)
        # Convert complex coefficients to real and imaginary
        coefficients = torch.view_as_real(coefficients)
        # Place real and imaginary coefficients under channel dimension
        coefficients = coefficients.transpose(-1, -2).transpose(-2, -3)

        return coefficients

    @staticmethod
    def to_complex(coefficients):
        """
        Convert a set of real coefficients to their equivalent complex representation.

        Parameters
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of real/imaginary CQT coefficients

        Returns
        ----------
        coefficients : Tensor (B x F X T)
          Batch of complex CQT coefficients
        """

        # Move real and imaginary coefficients to last dimension
        coefficients = coefficients.transpose(-3, -2).transpose(-2, -1)
        # Convert real and imaginary coefficients to complex
        coefficients = torch.view_as_complex(coefficients.contiguous())

        return coefficients

    @staticmethod
    def to_magnitude(coefficients):
        """
        Compute the magnitude for a set of real coefficients.

        Parameters
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of real/imaginary CQT coefficients

        Returns
        ----------
        magnitude : Tensor (B x F X T)
          Batch of magnitude coefficients
        """

        # Compute L2-norm of coefficients to compute magnitude
        magnitude = coefficients.norm(p=2, dim=-3)

        return magnitude

    @staticmethod
    def to_decibels(magnitude, rescale=True):
        """
        Convert a set of magnitude coefficients to decibels.

        TODO - move 0 dB only if maximum is higher?
             - currently it's consistent with previous dB scaling

        Parameters
        ----------
        magnitude : Tensor (B x F X T)
          Batch of magnitude coefficients (amplitude)
        rescale : bool
          Rescale decibels to the range [0, 1]

        Returns
        ----------
        decibels : Tensor (B x F X T)
          Batch of magnitude coefficients (dB)
        """

        decibels = list()

        for m in magnitude:
            # Convert to decibels
            d = AmplitudeToDB(stype='amplitude', top_db=80)(m)

            if rescale:
                # Make 0 dB ceiling
                d -= d.max()
                # Rescale decibels to range [0, 1]
                d = 1 + d / 80

            # Add converted sample to list
            decibels.append(d.unsqueeze(0))

        # Reconstruct original batch
        decibels = torch.cat(decibels, dim=0)

        return decibels

    def decode(self, coefficients):
        """
        Invert CQT spectral coefficients to synthesize audio.

        Parameters
        ----------
        coefficients : Tensor (B x 2 OR 1 x F X T)
          Batch of real/imaginary OR complex CQT coefficients

        Returns
        ----------
        output : Tensor (B x 1 x T)
          Batch of reconstructed audio
        """

        with torch.no_grad():
            if not coefficients.is_complex():
                # Convert real coefficients to complex representation
                coefficients = self.to_complex(coefficients)
                # Add a channel dimension to coefficients
                coefficients = coefficients.unsqueeze(-3)

            # Decode the complex CQT coefficients
            audio = super().decode(coefficients)

        return audio

    def pad_to_block_length(self, audio):
        """
        Pad audio to the next multiple of block length such that it can be processed in full.

        Parameters
        ----------
        audio : Tensor (B x 1 X T)
          Batch of audio

        Returns
        ----------
        audio : Tensor (B x 1 X T + p)
          Batch of padded audio
        """

        # Pad the audio with zeros to fill up the remainder of the final block
        audio = torch.nn.functional.pad(audio, (0, -audio.size(-1) % self.block_length))

        return audio

    def get_expected_samples(self, t):
        """
        Determine the number of samples corresponding to a specified amount of time.

        Parameters
        ----------
        t : float
          Amount of time

        Returns
        ----------
        num_samples : int
          Number of audio samples expected
        """

        # Compute number of samples and round down
        num_samples = int(max(0, t) * self.sample_rate)

        return num_samples

    def get_expected_frames(self, num_samples):
        """
        Determine the number of frames the module will return for a given number of samples.

        Parameters
        ----------
        num_samples : int
          Number of audio samples available

        Returns
        ----------
        num_frames : int
          Number of frames expected
        """

        # Number frames of coefficients per chunk times amount of chunks
        num_frames = math.ceil((num_samples / self.block_length) * self.max_window_length)

        return num_frames

    def get_times(self, n_frames):
        """
        Determine the time associated with each frame of coefficients.

        Parameters
        ----------
        n_frames : int
          Number of frames available

        Returns
        ----------
        times : ndarray (T)
          Time (seconds) associated with each frame
        """

        # Compute times as cumulative hops in seconds
        times = np.arange(n_frames) * self.hop_length / self.sample_rate

        return times

    def get_midi_freqs(self):
        """
        Obtain the MIDI frequencies associated with each bin.

        Returns
        ----------
        midi_freqs : ndarray (F)
          Center frequency of each bin
        """

        # Access pre-existing field
        midi_freqs = self.midi_freqs

        return midi_freqs
