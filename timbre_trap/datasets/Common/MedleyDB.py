from .. import BaseDataset

import yaml
import os


class MedleyDB(BaseDataset):
    """
    Implements the top-level wrapper for the MedleyDB dataset (https://medleydb.weebly.com/).
    """

    def __init__(self, **kwargs):
        """
        Add a field to store metadata for all available multitracks.
        """

        # Initialize dictionary for all metadata
        self.metadata = None

        BaseDataset.__init__(self, **kwargs)

    def load_metadata(self):
        """
        Load and process all metadata.
        """

        # Initialize dictionary for all metadata
        self.metadata = dict()

        for multitrack in self.available_multitracks():
            # Construct a path to the YAML-encoded metadata
            yaml_path = os.path.join(self.base_dir, 'Metadata', f'{multitrack}_METADATA.yaml')

            with open(yaml_path, 'r') as file:
                # Load the multitrack metadata into the dictionary
                self.metadata[multitrack] = yaml.safe_load(file)

    def available_multitracks(self):
        """
        Get the names of all available multitracks in the dataset.

        Returns
        ----------
        multitracks : list of strings
          List containing song (multitrack) names
        """

        # Construct a path to the directory containing audio
        audio_dir = os.path.join(self.base_dir, 'Audio')

        # Obtain a list of all multitracks for which audio is available
        multitracks = [d for d in os.listdir(audio_dir)
                       if os.path.isdir(os.path.join(audio_dir, d))
                       and not d.startswith('Bach10')]

        return multitracks

    @staticmethod
    def available_genres():
        """
        Obtain a list of genres in the dataset.

        Returns
        ----------
        genres : list of strings
          Genres of the original songs
        """

        genres = ['Classical',
                  'Electronic/Fusion',
                  'Jazz',
                  'Musical Theatre',
                  'Pop',
                  'Rap',
                  'Rock',
                  'Singer/Songwriter',
                  'World/Folk']

        return genres

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
            'violin section',
            'viola section',
            'cello section',
            'string section',
            'dilruba', # not in V1/V2
        # strings - plucked
            'acoustic guitar',
            'banjo',
            'guzheng',
            'harp',
            'harpsichord', # not in MedleyDB
            'liuqin',
            'mandolin',
            'oud',
            'slide guitar', # not in MedleyDB
            'ukulele', # not in MedleyDB
            'zhongruan',
            'sitar', # not in V1/V2
        # strings - struck
            'dulcimer', # not in MedleyDB
            'yangqin',
            'piano',
            'tack piano',
        # winds - flutes
            'dizi',
            'flute',
            'flute section',
            'piccolo',
            'bamboo flute',
            'panpipes', # not in MedleyDB
            'recorder', # not in MedleyDB
        # winds - single reeds
            'alto saxophone',
            'baritone saxophone',
            'bass clarinet',
            'clarinet',
            'clarinet section',
            'tenor saxophone',
            'soprano saxophone',
        # winds - double reeds
            'oboe',
            'english horn', # not in MedleyDB
            'bassoon',
            'bagpipe', # not in MedleyDB
        # winds - brass
            'trumpet',
            'cornet',
            'trombone',
            'french horn',
            'euphonium',
            'tuba',
            'brass section',
            'french horn section',
            'trombone section',
            'horn section',
            'trumpet section',
        # winds - free reeds
            'harmonica',
            'concertina', # not in MedleyDB
            'accordion',
            'bandoneon', # not in MedleyDB
            'harmonium', # not in MedleyDB
            'pipe organ', # not in MedleyDB
            'melodica',
        # voices
            'male singer',
            'female singer',
            'male speaker',
            'female speaker', # not in V1/V2
            'male rapper',
            'female rapper', # not in MedleyDB
            'beatboxing', # not in MedleyDB
            'vocalists',
            'choir', # not in V1/V2
            'crowd', # not in V1/V2
            'male screamer', # not in V1/V2
            'female screamer', # not in MedleyDB
        # percussion - idiophones
            'triangle', # not in MedleyDB
            'sleigh bells',
            'cowbell',
            'cabasa',
            'high hat',
            'gong',
            'guiro',
            'gu',
            'cymbal',
            'chimes',
            'castanet',
            'claps',
            'rattle', # not in MedleyDB
            'shaker',
            'maracas', # not in MedleyDB
            'xylophone', # not in MedleyDB
            'vibraphone',
            'marimba', # not in MedleyDB
            'glockenspiel',
            'whistle',
            'snaps', # not in V1/V2
        # percussion - drums
            'timpani',
            'toms',
            'snare drum',
            'kick drum',
            'bass drum',
            'bongo',
            'conga', # not in MedleyDB
            'tambourine',
            'darbuka',
            'doumbek',
            'tabla',
            'auxiliary percussion',
            'drum set',
        # electric - amplified
            'clean electric guitar',
            'distorted electric guitar',
            'electric bass',
            'lap steel guitar',
        # electric - electronic
            'drum machine',
            'electric piano',
            'electronic organ',
            'synthesizer',
            'theremin', # not in MedleyDB
            'fx/processed sound',
            'scratches',
            'sampler',
        # other
            'Main System',
            'Unlabeled', # not in V1/V2
            'woodwind section' # not included in original taxonomy, but exists in Black Mirror - Arcade Fire (not in V1/V2)
        ]

        return instruments

    @classmethod
    def download(cls, save_dir):
        """
        At this time, you must request access for MedleyDB and
        it must be downloaded manually, so an error is thrown.

        Downloads:
        - Audio (V1): https://zenodo.org/record/1649325
        - Audio (V2): https://zenodo.org/record/1715175
        - Annotations: https://github.com/marl/medleydb/tree/master/medleydb/data/Annotations
        - Metadata: https://github.com/marl/medleydb/tree/master/medleydb/data/Metadata
        - Pitch Tracking Subset: https://zenodo.org/record/2620624
        - Re-Synthesized Stems: https://zenodo.org/record/1481172

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of MedleyDB
        """

        return NotImplementedError
