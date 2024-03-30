from ..Common import MedleyDB

import os


class MedleyDB(MedleyDB):
    """
    Implements a wrapper for the MedleyDB dataset to analyze the stems individually.
    """

    @staticmethod
    def available_instruments():
        """
        Obtain a subset of pitched instruments in the dataset.

        See https://github.com/marl/medleydb/blob/master/medleydb/resources/taxonomy.yaml
        and https://github.com/marl/medleydb/blob/master/medleydb/resources/instrument_f0_type.json

        Returns
        ----------
        instruments : list of strings
          Pitched instruments played within stems
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
            'vocalists',
            'choir', # not in V1/V2
            'crowd', # not in V1/V2
            'male screamer', # not in V1/V2
            'female screamer', # not in MedleyDB
        # percussion - idiophones
            'chimes',
            'xylophone', # not in MedleyDB
            'vibraphone',
            'marimba', # not in MedleyDB
            'glockenspiel',
            'whistle',
        # percussion - drums
            'timpani',
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
            'sampler',
        # other
            'woodwind section' # not included in original taxonomy, but exists in Black Mirror - Arcade Fire (not in V1/V2)
        ]

        return instruments

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Solo instruments within the collection of stems
        """

        splits = MedleyDB.available_instruments()

        return splits

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

        # Initialize a list to hold valid tracks
        tracks = list()

        for multitrack in self.metadata.keys():
            # Loop through all stems of the mixture
            for stem in self.metadata[multitrack]['stems'].values():
                # Loop through all raw stem components
                for raw_audio in stem['raw'].values():
                    # Check if raw stem represents specified instrument
                    if split == raw_audio['instrument']:
                        # Add the track name as the top-level audio directory with the raw audio file name
                        tracks.append(os.path.join(multitrack, os.path.splitext(raw_audio['filename'])[0]))

        return tracks

    def get_audio_path(self, track):
        """
        Get the path to a track's audio.

        Parameters
        ----------
        track : string
          MedleyDB track name

        Returns
        ----------
        wav_path : string
          Path to audio for the specified track
        """

        # Break apart the track name
        mixture, stem = os.path.split(track)

        # Construct the path to the audio stem
        wav_path = os.path.join(self.base_dir, 'Audio', mixture, f'{mixture}_RAW', f'{stem}.wav')

        return wav_path
