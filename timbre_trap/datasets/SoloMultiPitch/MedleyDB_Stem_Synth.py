from .. import BaseDataset
from ..Common import MedleyDB
from . import MedleyDB_Pitch
from ..utils import *

import shutil
import os


class MedleyDB_Stem_Synth(MedleyDB_Pitch):
    """
    Implements a wrapper for the Re-Synthesized (Perfect F0) MedleyDB Subset to analyze the stems.
    """

    def load_metadata(self):
        """
        Call the top-level MedleyDB wrapper's function and remove unavailable stems.
        """

        # Load full MedleyDB metadata
        MedleyDB.load_metadata(self)

        # Loop through all metadata entries
        for multitrack in self.metadata.keys():
            # Determine which stems are available for the multitrack
            stems = [s.split('_STEM_')[-1].split('.')[0]
                     for s in self.available_stems()
                     if s.startswith(multitrack)]

            # Obtain a list of all the stems within the multitrack
            all_stems = list(self.metadata[multitrack]['stems'].keys())

            for k in all_stems:
                # Check stem availability
                if k[1:] not in stems:
                    # Remove metadata for unavailable stems
                    self.metadata[multitrack]['stems'].pop(k)

    def available_stems(self):
        """
        Get the names of all stems in the subset.

        Returns
        ----------
        stems : list of strings
          List containing stem names
        """

        # Construct a path to the directory containing pitch annotations
        annotation_dir = os.path.join(self.base_dir, 'annotation_stems')

        # Obtain a list of the stem names within the annotation directory
        stems = [f for f in os.listdir(annotation_dir) if f.endswith('.csv')]

        return stems

    def available_multitracks(self):
        """
        Get the names of all originating multitracks for the stems in the dataset.

        Returns
        ----------
        multitracks : list of strings
          List containing song (multitrack) names
        """

        # Obtain a list of the stem names
        stems = self.available_stems()

        # Obtain a list of all unique multitracks from the stem names
        multitracks = sorted(set([anno.split('_STEM_')[0] for anno in stems]))

        return multitracks

    @staticmethod
    def available_instruments():
        """
        Obtain a list of instruments in the dataset.

        See https://github.com/marl/medleydb/blob/master/medleydb/resources/taxonomy.yaml

        Returns
        ----------
        instruments : list of strings
          Instruments played within stems
        """

        instruments = [
        # strings - bowed
            'erhu',
            'violin',
            'viola',
            'cello',
            'double bass',
        # strings - plucked
            # None
        # strings - struck
            # None
        # winds - flutes
            'dizi',
            'flute',
            'piccolo',
            'bamboo flute',
        # winds - single reeds
            'alto saxophone',
            'baritone saxophone',
            'bass clarinet',
            'clarinet',
            'tenor saxophone',
            'soprano saxophone',
        # winds - double reeds
            'oboe',
            'bassoon',
        # winds - brass
            'trumpet',
            'trombone',
            'french horn',
            'tuba',
        # winds - free reeds
            # None
        # voices
            'male singer',
            'female singer',
            'male rapper',
        # percussion - idiophones
            # None
        # percussion - drums
            # None
        # electric - amplified
            'electric bass',
        # electric - electronic
            # None
        # other
            # None
        ]

        return instruments

    def get_tracks(self, split):
        """
        Get the names of the tracks in the dataset.

        Parameters
        ----------
        split : string
          Instrument identifier

        Returns
        ----------
        tracks : list of strings
          List containing the stems with the specified instrument
        """

        if self.metadata is None:
            # Load metadata only once
            self.load_metadata()

        # Initialize a list to hold valid tracks
        tracks = list()

        for multitrack in self.metadata.keys():
            # Loop through all stems of the mixture
            for stem in self.metadata[multitrack]['stems'].values():
                # Check if stem contains specified instrument
                if split == stem['instrument']:
                    # Add the stem name to the list of tracks
                    tracks.append(os.path.splitext(stem['filename'])[0])

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MedleyDB-Stem-Synth track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Get the path to the audio stem
        wav_path = os.path.join(self.base_dir, 'audio_stems', f'{track}.RESYN.wav')

        return wav_path

    def get_ground_truth_path(self, track):
        """
        Get the path to a track's ground-truth.

        Parameters
        ----------
        track : string
          MedleyDB-Stem-Synth track name

        Returns
        ----------
        csv_path : string
          Path to ground-truth for the specified track
        """

        # Get the path to the F0 annotations for the stem
        csv_path = os.path.join(self.base_dir, 'annotation_stems', f'{track}.RESYN.csv')

        return csv_path

    @classmethod
    def download(cls, save_dir):
        """
        Download the MedleyDB-Stem-Synth dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MedleyDB-Stem-Synth
        """

        # Create top-level directory
        BaseDataset.download(save_dir)

        # URL pointing to the MedleyDB repository on GitHub
        mdb_url = 'https://github.com/marl/medleydb/archive/refs/heads/master.zip'

        # Construct a path for saving the MedleyDB repository
        mdb_path = os.path.join(save_dir, os.path.basename(mdb_url))

        # Download MedleyDB repository
        stream_url_resource(mdb_url, mdb_path, 1000 * 1024)

        # Unzip the repository and remove the zip file
        unzip_and_remove(mdb_path)

        # Directory of the repository after unzipping
        repo_dir = os.path.join(save_dir, 'medleydb-master')

        # Directory to hold metadata for MedleyDB multitracks
        metadata_dir = os.path.join(save_dir, 'Metadata')

        # Create metadata directory
        os.makedirs(metadata_dir)

        # Move the folder containing metadata from the unzipped repository to the base directory
        change_base_dir(metadata_dir, os.path.join(repo_dir, 'medleydb', 'data', 'Metadata'))

        # Remove the remainder of the repository
        shutil.rmtree(os.path.join(save_dir, 'medleydb-master'))

        # URL pointing to the tar file containing data for all tracks
        tar_url = 'https://zenodo.org/record/1481172/files/MDB-stem-synth.tar.gz'

        # Construct a path for saving the tar file
        tar_path = os.path.join(save_dir, os.path.basename(tar_url))

        # Download the tar file
        stream_url_resource(tar_url, tar_path, 1000 * 1024)

        # Untar the downloaded file and remove it
        unzip_and_remove(tar_path, tar=True)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'MDB-stem-synth'))
