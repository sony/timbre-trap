from . import CQT

from tqdm import tqdm

import torch.nn as nn
import torch


__all__ = [
    'TimbreTrap',
    'Encoder',
    'Decoder',
    'EncoderBlock',
    'DecoderBlock',
    'ResidualConv2dBlock',
    'TimbreTrapFiLM',
    'FiLM',
    'TimbreTrapMag',
    'TimbreTrapMagDB'
]


class TimbreTrap(nn.Module):
    """
    Implements a 2D convolutional U-Net architecture based loosely on SoundStream.
    """

    def __init__(self, sample_rate, n_octaves, bins_per_octave, secs_per_block=3,
                       latent_size=None, model_complexity=1, skip_connections=False):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        sample_rate : int
          Expected sample rate of input
        n_octaves : int
          Number of octaves below Nyquist frequency to represent
        bins_per_octave : int
          Number of frequency bins within each octave
        secs_per_block : float
          Number of seconds to process at once with sliCQ
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters and embedding sizes
        skip_connections : bool
          Whether to include skip connections between encoder and decoder
        """

        nn.Module.__init__(self)

        self.sliCQ = CQT(n_octaves=n_octaves,
                         bins_per_octave=bins_per_octave,
                         sample_rate=sample_rate,
                         secs_per_block=secs_per_block)

        self.encoder = Encoder(feature_size=self.sliCQ.n_bins, latent_size=latent_size, model_complexity=model_complexity)
        self.decoder = Decoder(feature_size=self.sliCQ.n_bins, latent_size=latent_size, model_complexity=model_complexity)

        if skip_connections:
            # Start by adding encoder features with identity weighting
            self.skip_weights = torch.nn.Parameter(torch.ones(5))
        else:
            # No skip connections
            self.skip_weights = None

    def encode(self, audio):
        """
        Encode a batch of raw audio into latent codes.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Batch of input raw audio

        Returns
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)]
          Embeddings produced by encoder at each level
        losses : dict containing
          ...
        """

        # Compute CQT spectral features
        coefficients = self.sliCQ(audio)

        # Encode features into latent vectors
        latents, embeddings, losses = self.encoder(coefficients)

        return latents, embeddings, losses

    def apply_skip_connections(self, embeddings):
        """
        Apply skip connections to encoder embeddings, or discard the embeddings if skip connections do not exist.

        Parameters
        ----------
        embeddings : list of [Tensor (B x C x E x T)]
          Embeddings produced by encoder at each level

        Returns
        ----------
        embeddings : list of [Tensor (B x C x E x T)]
          Encoder embeddings scaled with learnable weight
        """

        if self.skip_weights is not None:
            # Apply a learnable weight to the embeddings for the skip connection
            embeddings = [self.skip_weights[i] * e for i, e in enumerate(embeddings)]
        else:
            # Discard embeddings from encoder
            embeddings = None

        return embeddings

    def decode(self, latents, embeddings=None, transcribe=False):
        """
        Decode a batch of latent codes into logits representing real/imaginary coefficients.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of output logits [-∞, ∞]
        """

        # Create binary values to indicate function decoder should perform
        indicator = (not transcribe) * torch.ones_like(latents[..., :1, :])

        # Concatenate indicator to final dimension of latents
        latents = torch.cat((latents, indicator), dim=-2)

        # Decode latent vectors into real/imaginary coefficients
        coefficients = self.decoder(latents, embeddings)

        return coefficients

    def _inference(self, audio, transcribe=False):
        """
        Encode audio into latents, then decode into
        transcription or reconstruction coefficients.

        Parameters
        ----------
        audio : Tensor (B x 1 x N)
          Batch of input raw audio
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of spectral coefficients [-∞, ∞]
        """

        # Encode raw audio into latent vectors
        latents, embeddings, _ = self.encode(audio)

        # Apply skip connections if they are turned on
        embeddings = self.apply_skip_connections(embeddings)

        # Obtain coefficients with appropriate switch setting
        coefficients = self.decode(latents, embeddings, transcribe)

        return coefficients

    def inference(self, audio, transcribe=False):
        """
        Perform full-length inference on a batch of raw audio.

        Parameters
        ----------
        audio : Tensor (B x 1 x N)
          Batch of input raw audio
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of spectral coefficients [-∞, ∞]
        """

        # Pad audio to next multiple of block length
        audio = self.sliCQ.pad_to_block_length(audio)

        with torch.no_grad():
            # Perform inference on the full-length audio
            coefficients = self._inference(audio, transcribe)

        return coefficients

    def chunked_inference(self, audio, transcribe=False):
        """
        Perform inference iteratively on a batch of raw audio.

        Parameters
        ----------
        audio : Tensor (B x 1 x N)
          Batch of input raw audio
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of spectral coefficients [-∞, ∞]
        """

        # Determine appropriate device
        device = audio.device
        # Determine batch size and bin length
        B, F = audio.size(0), self.sliCQ.n_bins

        # Pad audio to next multiple of block length
        audio = self.sliCQ.pad_to_block_length(audio)

        with torch.no_grad():
            # Compute hop length for 50% overlap
            hop_length = self.sliCQ.block_length // 2
            # Pad both sides of audio to center first and last block
            audio = torch.nn.functional.pad(audio, [hop_length] * 2)
            # Determine total number of chunks to process
            n_chunks = (audio.size(-1) - hop_length) // hop_length

            # Determine number of frames for each chunk
            n_frames_chunk = self.sliCQ.max_window_length
            # Initialize a function for windowing chunked output
            window = torch.signal.windows.hann(n_frames_chunk, device=device)

            # Determine total number of frames corresponding to audio
            n_frames = self.sliCQ.get_expected_frames(audio.size(-1))
            # Initialize a Tensor of zeros for final output coefficients
            coefficients = torch.zeros((B, 2, F, n_frames), device=device)

            # Process chunks of audio iteratively and display a progress bar
            for i in tqdm(range(n_chunks), position=0, leave=True, desc='\t\tprocessing chunks'):
                # Compute sample boundaries for slice
                sample_start = i * hop_length
                sample_stop = sample_start + self.sliCQ.block_length

                # Slice the next chunk of audio to block length
                audio_chunk = audio[..., sample_start : sample_stop]

                # Perform inference on the current chunk of audio
                output_chunk = self._inference(audio_chunk, transcribe)

                # Compute sample boundaries for slice
                frame_start = i * n_frames_chunk // 2
                frame_stop = frame_start + n_frames_chunk

                # Apply windowing and add chunk output to final output coefficients
                coefficients[..., frame_start : frame_stop] += window * output_chunk

            # Remove output frames corresponding to extra padding
            coefficients = coefficients[..., n_frames_chunk // 2 :
                                             -n_frames_chunk // 2]

        return coefficients

    def to_activations(self, coefficients):
        """
        Obtain activations for a batch of transcription coefficients.

        Parameters
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of transcription coefficients [-∞, ∞]

        Returns
        ----------
        activations : Tensor (B x F X T)
          Batch of multi-pitch activations [0, 1]
        """

        # Extract magnitude of decoded coefficients and compress to [0, 1] range
        activations = torch.nn.functional.tanh(self.sliCQ.to_magnitude(coefficients))

        return activations


    def transcribe(self, audio):
        """
        Obtain multi-pitch activations for a batch of raw audio.

        Parameters
        ----------
        audio : Tensor (B x 1 x N)
          Batch of input raw audio

        Returns
        ----------
        activations : Tensor (B x F X T)
          Batch of multi-pitch activations [0, 1]
        """

        # Obtain transcription coefficients for the audio
        coefficients = self.chunked_inference(audio, True)

        # Convert transcription coefficients into activations
        activations = self.to_activations(coefficients)

        return activations

    def reconstruct(self, audio_in):
        """
        Encode and reconstruct a batch of raw audio.

        Parameters
        ----------
        audio_in : Tensor (B x 1 x N)
          Batch of input raw audio

        Returns
        ----------
        audio_out : Tensor (B x 1 x L)
          Batch of reconstructed audio
        """

        # Obtain reconstructed spectral coefficients for the audio
        coefficients = self.chunked_inference(audio_in, False)

        # Decode coefficients for reconstructed audio
        audio_out = self.sliCQ.decode(coefficients)

        return audio_out

    def forward(self, audio, consistency=False):
        """
        Perform all model functions efficiently (for training/evaluation).

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Batch of input raw audio
        consistency : bool
          Whether to perform computations for consistency loss

        Returns
        ----------
        reconstruction : Tensor (B x 2 x F X T)
          Batch of reconstructed spectral coefficients
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        transcription : Tensor (B x 2 x F X T)
          Batch of transcription coefficients
        transcription_rec : Tensor (B x 2 x F X T)
          Batch of reconstructed coefficients for transcription
        transcription_scr : Tensor (B x 2 x F X T)
          Batch of transcription coefficients for transcription
        losses : dict containing
          ...
        """

        # Encode raw audio into latent vectors
        latents, embeddings, losses = self.encode(audio)

        # Apply skip connections if they are turned on
        embeddings = self.apply_skip_connections(embeddings)

        # Decode latent vectors into spectral coefficients
        reconstruction = self.decode(latents, embeddings)

        # Obtain coefficients using transcription switch
        transcription = self.decode(latents, embeddings, True)

        if consistency:
            # Re-encode transcription coefficients
            latents_trn, embeddings_trn, _ = self.encoder(transcription)

            # Apply skip connections if they are turned on
            embeddings_trn = self.apply_skip_connections(embeddings_trn)

            # Attempt to reconstruct transcription coefficients
            transcription_rec = self.decode(latents_trn, embeddings_trn)

            # Attempt to transcribe audio pertaining to transcription coefficients
            transcription_scr = self.decode(latents_trn, embeddings_trn, True)
        else:
            # Return null for both sets of coefficients
            transcription_rec, transcription_scr = None, None

        return reconstruction, latents, transcription, transcription_rec, transcription_scr, losses


class Encoder(nn.Module):
    """
    Implements a 2D convolutional encoder.
    """

    def __init__(self, feature_size, latent_size=None, model_complexity=1):
        """
        Initialize the encoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (2  * 2 ** (model_complexity - 1),
                    4  * 2 ** (model_complexity - 1),
                    8  * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    32 * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = 32 * 2 ** (model_complexity - 1)
              
        self.convin = nn.Sequential(
            nn.Conv2d(2, channels[0], kernel_size=3, padding='same'),
            nn.ELU(inplace=True)
        )

        self.block1 = EncoderBlock(channels[0], channels[1], stride=2)
        self.block2 = EncoderBlock(channels[1], channels[2], stride=2)
        self.block3 = EncoderBlock(channels[2], channels[3], stride=2)
        self.block4 = EncoderBlock(channels[3], channels[4], stride=2)

        embedding_size = feature_size

        for i in range(4):
            # Dimensionality after strided convolutions
            embedding_size = embedding_size // 2 - 1

        self.convlat = nn.Conv2d(channels[4], latent_size, kernel_size=(embedding_size, 1))

    def forward(self, coefficients):
        """
        Encode a batch of input spectral coefficients.

        Parameters
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of spectral coefficients

        Returns
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)]
          Embeddings produced by encoder at each level
        losses : dict containing
          ...
        """

        # Initialize a list to hold features for skip connections
        embeddings = list()

        # Encode features into embeddings
        embeddings.append(self.convin(coefficients))
        embeddings.append(self.block1(embeddings[-1]))
        embeddings.append(self.block2(embeddings[-1]))
        embeddings.append(self.block3(embeddings[-1]))
        embeddings.append(self.block4(embeddings[-1]))

        # Compute latent vectors from embeddings
        latents = self.convlat(embeddings[-1]).squeeze(-2)

        # No encoder losses
        loss = dict()

        return latents, embeddings, loss


class Decoder(nn.Module):
    """
    Implements a 2D convolutional decoder.
    """

    def __init__(self, feature_size, latent_size=None, model_complexity=1):
        """
        Initialize the decoder.

        Parameters
        ----------
        feature_size : int
          Dimensionality of input features
        latent_size : int or None (Optional)
          Dimensionality of latent space
        model_complexity : int
          Scaling factor for number of filters
        """

        nn.Module.__init__(self)

        channels = (32 * 2 ** (model_complexity - 1),
                    16 * 2 ** (model_complexity - 1),
                    8  * 2 ** (model_complexity - 1),
                    4  * 2 ** (model_complexity - 1),
                    2  * 2 ** (model_complexity - 1))

        # Make sure all channel sizes are integers
        channels = tuple([round(c) for c in channels])

        if latent_size is None:
            # Set default dimensionality
            latent_size = 32 * 2 ** (model_complexity - 1)

        padding = list()

        embedding_size = feature_size

        for i in range(4):
            # Padding required for expected output size
            padding.append(embedding_size % 2)
            # Dimensionality after strided convolutions
            embedding_size = embedding_size // 2 - 1

        # Reverse order
        padding.reverse()

        self.convin = nn.Sequential(
            nn.ConvTranspose2d(latent_size + 1, channels[0], kernel_size=(embedding_size, 1)),
            nn.ELU(inplace=True)
        )

        self.block1 = DecoderBlock(channels[0], channels[1], stride=2, padding=padding[0])
        self.block2 = DecoderBlock(channels[1], channels[2], stride=2, padding=padding[1])
        self.block3 = DecoderBlock(channels[2], channels[3], stride=2, padding=padding[2])
        self.block4 = DecoderBlock(channels[3], channels[4], stride=2, padding=padding[3])

        self.convout = nn.Conv2d(channels[4], 2, kernel_size=3, padding='same')

    def forward(self, latents, encoder_embeddings=None):
        """
        Decode a batch of input latent codes.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        encoder_embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level

        Returns
        ----------
        output : Tensor (B x 2 x F X T)
          Batch of output logits [-∞, ∞]
        """

        # Restore feature dimension
        latents = latents.unsqueeze(-2)

        # Process latents with decoder blocks
        embeddings = self.convin(latents)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-1]

        embeddings = self.block1(embeddings)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-2]

        embeddings = self.block2(embeddings)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-3]

        embeddings = self.block3(embeddings)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-4]

        embeddings = self.block4(embeddings)

        if encoder_embeddings is not None:
            embeddings = embeddings + encoder_embeddings[-5]

        # Decode embeddings into spectral logits
        output = self.convout(embeddings)

        return output


class EncoderBlock(nn.Module):
    """
    Implements a chain of residual convolutional blocks with progressively
    increased dilation, followed by down-sampling via strided convolution.
    """

    def __init__(self, in_channels, out_channels, stride=2):
        """
        Initialize the encoder block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        stride : int
          Stride for the final convolutional layer
        """

        nn.Module.__init__(self)

        self.block1 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=1)
        self.block2 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=2)
        self.block3 = ResidualConv2dBlock(in_channels, in_channels, kernel_size=3, dilation=3)

        self.hop = stride
        self.win = 2 * stride

        self.sconv = nn.Sequential(
            # Down-sample along frequency (height) dimension via strided convolution
            nn.Conv2d(in_channels, out_channels, kernel_size=(self.win, 1), stride=(self.hop, 1)),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        """
        Feed features through the encoder block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Process features
        y = self.block1(x)
        y = self.block2(y)
        y = self.block3(y)

        # Down-sample
        y = self.sconv(y)

        return y


class DecoderBlock(nn.Module):
    """
    Implements up-sampling via transposed convolution, followed by a chain
    of residual convolutional blocks with progressively increased dilation.
    """

    def __init__(self, in_channels, out_channels, stride=2, padding=0):
        """
        Initialize the encoder block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        stride : int
          Stride for the transposed convolution
        padding : int
          Number of features to pad after up-sampling
        """

        nn.Module.__init__(self)

        self.hop = stride
        self.win = 2 * stride

        self.tconv = nn.Sequential(
            # Up-sample along frequency (height) dimension via transposed convolution
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=(self.win, 1), stride=(self.hop, 1), output_padding=(padding, 0)),
            nn.ELU(inplace=True)
        )

        self.block1 = ResidualConv2dBlock(out_channels, out_channels, kernel_size=3, dilation=1)
        self.block2 = ResidualConv2dBlock(out_channels, out_channels, kernel_size=3, dilation=2)
        self.block3 = ResidualConv2dBlock(out_channels, out_channels, kernel_size=3, dilation=3)

    def forward(self, x):
        """
        Feed features through the decoder block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Up-sample
        y = self.tconv(x)

        # Process features
        y = self.block1(y)
        y = self.block2(y)
        y = self.block3(y)

        return y


class ResidualConv2dBlock(nn.Module):
    """
    Implements a 2D convolutional block with dilation, no down-sampling, and a residual connection.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, dilation=1):
        """
        Initialize the convolutional block.

        Parameters
        ----------
        in_channels : int
          Number of input feature channels
        out_channels : int
          Number of output feature channels
        kernel_size : int
          Kernel size for convolutions
        dilation : int
          Amount of dilation for first convolution
        """

        nn.Module.__init__(self)

        self.conv1 = nn.Sequential(
            # TODO - only dilate across frequency?
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding='same', dilation=dilation),
            nn.ELU(inplace=True)
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=1),
            nn.ELU(inplace=True)
        )

    def forward(self, x):
        """
        Feed features through the convolutional block.

        Parameters
        ----------
        x : Tensor (B x C_in x H x W)
          Batch of input features

        Returns
        ----------
        y : Tensor (B x C_out x H x W)
          Batch of corresponding output features
        """

        # Process features
        y = self.conv1(x)
        y = self.conv2(y)

        # Residual connection
        y = y + x

        return y


class TimbreTrapFiLM(TimbreTrap):
    """
    Variant of autoencoder with FiLM layer before decoder.
    """

    def __init__(self, sample_rate, n_octaves, bins_per_octave, secs_per_block=3,
                       latent_size=None, model_complexity=1, skip_connections=False):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        See TimbreTrap class...
        """

        TimbreTrap.__init__(self, sample_rate, n_octaves, bins_per_octave, secs_per_block,
                                  latent_size, model_complexity, skip_connections)

        if latent_size is None:
            latent_size = self.decoder.convin.in_channels - 1

        convin_out_channels = self.decoder.convin[0].out_channels
        convin_kernel_size = self.decoder.convin[0].kernel_size

        self.decoder.convin = nn.Sequential(
            nn.ConvTranspose2d(latent_size, convin_out_channels, kernel_size=convin_kernel_size),
            nn.ELU(inplace=True)
        )

        # Initialize the FiLM conditioning layer
        self.film_layer = FiLM(latent_size, n_conditions=2)

    def decode(self, latents, embeddings=None, transcribe=False):
        """
        Decode a batch of latent codes into logits representing real/imaginary coefficients.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 2 x F X T)
          Batch of output logits [-∞, ∞]
        """

        # Create a one-hot vector indicating which function to perform [transcription, reconstruction]
        condition = torch.tensor([transcribe, not transcribe], dtype=latents.dtype, device=latents.device)
        # Process latent vector with FiLM layer
        latents = self.film_layer(latents, condition)

        # Decode latent vectors into real/imaginary coefficients
        coefficients = self.decoder(latents, embeddings)

        return coefficients


class FiLM(nn.Module):
    """
    Implements an FiLM conditioning layer.
    """

    def __init__(self, embedding_size, n_conditions):
        """
        Initialize the layer.

        Parameters
        ----------
        embedding_size : int
          Dimensionality of input features
        n_conditions : int
          Number of conditions allowable
        """

        super().__init__()

        # Initialize linear layers
        self.gamma = nn.Linear(n_conditions, embedding_size)
        self.beta = nn.Linear(n_conditions, embedding_size)

    def forward(self, x, condition):
        """
        Feed features through the FiLM layer.

        Parameters
        ----------
        x : Tensor (B x D_lat x T)
          Batch of input embeddings
        condition : Tensor (n_conditions)
          One-hot vector indicating condition

        Returns
        ----------
        y : Tensor (B x D_lat x T)
          Batch of output embeddings
        """

        # Swap time and feature dimension
        x = x.transpose(-1, -2)
        # Apply gamma and beta to input embeddings
        y = x * self.gamma(condition) + self.beta(condition)
        # Restore original dimension order
        y = y.transpose(-1, -2)

        return y


class TimbreTrapMag(TimbreTrap):
    """
    Magnitude-CQT (amplitude) variant of autoencoder.
    """

    def __init__(self, sample_rate, n_octaves, bins_per_octave, secs_per_block=3,
                       latent_size=None, model_complexity=1, skip_connections=False):
        """
        Initialize the full autoencoder.

        Parameters
        ----------
        See TimbreTrap class...
        """

        TimbreTrap.__init__(self, sample_rate, n_octaves, bins_per_octave, secs_per_block,
                                  latent_size, model_complexity, skip_connections)

        convin_out_channels = self.encoder.convin[0].out_channels
        convout_in_channels = self.decoder.convout.in_channels

        self.encoder.convin = nn.Sequential(
            nn.Conv2d(1, convin_out_channels, kernel_size=3, padding='same'),
            nn.ELU(inplace=True)
        )

        self.decoder.convout = nn.Conv2d(convout_in_channels, 1, kernel_size=3, padding='same')

    def encode(self, audio):
        """
        Encode a batch of raw audio into latent codes.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Batch of input raw audio

        Returns
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)]
          Embeddings produced by encoder at each level
        losses : dict containing
          ...
        """

        # Compute CQT spectral features and convert to magnitude
        coefficients = self.sliCQ.to_magnitude(self.sliCQ(audio)).unsqueeze(-3)

        # Encode features into latent vectors
        latents, embeddings, losses = self.encoder(coefficients)

        return latents, embeddings, losses

    def decode(self, latents, embeddings=None, transcribe=False):
        """
        Decode a batch of latent codes into logits representing real/imaginary coefficients.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 1 x F X T)
          Batch of output logits [0, ∞]
        """

        # Perform standard decoding steps
        coefficients = super().decode(latents, embeddings, transcribe)

        # Make sure coefficients are non-negative
        coefficients = torch.nn.functional.relu(coefficients, inplace=True)

        return coefficients

    def to_activations(self, coefficients):
        """
        Obtain activations for a batch of transcription coefficients.

        Parameters
        ----------
        coefficients : Tensor (B x 1 x F X T)
          Batch of transcription magnitude coefficients [0, ∞]

        Returns
        ----------
        activations : Tensor (B x F X T)
          Batch of multi-pitch activations [0, 1]
        """

        # Remove channel dimension and convert logits to activations
        activations = torch.nn.functional.tanh(coefficients.squeeze(-3))

        return activations


class TimbreTrapMagDB(TimbreTrapMag):
    """
    Magnitude-CQT (decibels) variant of autoencoder.
    """

    def encode(self, audio):
        """
        Encode a batch of raw audio into latent codes.

        Parameters
        ----------
        audio : Tensor (B x 1 x T)
          Batch of input raw audio

        Returns
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)]
          Embeddings produced by encoder at each level
        losses : dict containing
          ...
        """

        # Compute CQT spectral features and convert to magnitude
        coefficients = self.sliCQ.to_magnitude(self.sliCQ(audio))

        # Convert magnitude to scaled decibels
        coefficients = self.sliCQ.to_decibels(coefficients).unsqueeze(-3)

        # Encode features into latent vectors
        latents, embeddings, losses = self.encoder(coefficients)

        return latents, embeddings, losses

    def decode(self, latents, embeddings=None, transcribe=False):
        """
        Decode a batch of latent codes into logits representing real/imaginary coefficients.

        Parameters
        ----------
        latents : Tensor (B x D_lat x T)
          Batch of latent codes
        embeddings : list of [Tensor (B x C x E x T)] or None (no skip connections)
          Embeddings produced by encoder at each level
        transcribe : bool
          Switch for transcription vs. reconstruction

        Returns
        ----------
        coefficients : Tensor (B x 1 x F X T)
          Batch of output logits [0, 1]
        """

        # Perform standard decoding steps
        coefficients = TimbreTrap.decode(self, latents, embeddings, transcribe)

        # Make sure coefficients are non-negative
        coefficients = torch.nn.functional.sigmoid(coefficients)

        return coefficients

    def to_activations(self, coefficients):
        """
        Obtain activations for a batch of transcription coefficients.

        Parameters
        ----------
        coefficients : Tensor (B x 1 x F X T)
          Batch of transcription magnitude coefficients [0, 1]

        Returns
        ----------
        activations : Tensor (B x F X T)
          Batch of multi-pitch activations [0, 1]
        """

        # Simply remove channel dimension
        activations = coefficients.squeeze(-3)

        return activations
