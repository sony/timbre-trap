from timbre_trap.utils.data import constants
from . import MPEDataset, NoteDataset

from abc import abstractmethod


class AMTDataset(MPEDataset, NoteDataset):
    """
    Implements functionality for a dataset with audio and note annotations.
    """

    @abstractmethod
    def __getitem__(self, index, offset_s=None):
        """
        Extract the audio and ground-truth data for a sampled track.

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
          times : Tensor (T)
            Time associated with each frame
          ground_truth : Tensor (F x T)
            Ground-truth activations for the track
        """

        # Determine corresponding track
        track = self.tracks[index]

        # Load the track's audio
        audio = self.get_audio(track)

        # Determine number of samples
        n_samples = audio.size(-1)

        # Pack the data into a dictionary
        data = {constants.KEY_TRACK : track,
                constants.KEY_AUDIO : audio}

        if self.n_secs is None:
            if self.cqt is not None:
                # Extract ground-truth for the full track
                data.update(NoteDataset.__getitem__(self, index, n_samples))
        else:
            # Randomly slice audio using default sequence length
            audio, offset_t = self.slice_audio(audio, offset_s=offset_s)

            # Add trimmed audio back to track data
            data.update({constants.KEY_AUDIO: audio})

            if self.cqt is not None:
                # Extract and slice ground-truth for the full track
                data.update(NoteDataset.__getitem__(self, index, n_samples, offset_t))

        return data
