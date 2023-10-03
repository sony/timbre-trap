from .. import AMTDataset
from ..utils import *

import os


class TRIOS(AMTDataset):
    """
    Implements the top-level wrapper for the TRIOS dataset (https://zenodo.org/record/6797837).
    """

    PITCHED_INSTRUMENTS = ['horn', 'piano', 'violin', 'bassoon', 'trumpet', 'clarinet', 'viola', 'cello', 'saxophone']

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Names of multitracks within dataset
        """

        splits = ['brahms', 'lussier', 'mozart', 'schubert', 'take_five']

        return splits

    @classmethod
    def download(cls, save_dir):
        """
        Download the TRIOS dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of TRIOS
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the zip file containing data for all tracks
        url = 'https://zenodo.org/record/6797837/files/TRIOS Dataset.zip'

        # Construct a path for saving the file
        zip_path = os.path.join(save_dir, os.path.basename(url))

        # Download the zip file
        stream_url_resource(url, zip_path, 1000 * 1024)

        # Unzip the downloaded file and remove it
        unzip_and_remove(zip_path)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'TRIOS Dataset'))
