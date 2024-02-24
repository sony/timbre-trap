from timbre_trap.utils.data import *
from .. import BaseDataset

import os


class URMP(BaseDataset):
    """
    Implements the top-level wrapper for the URMP dataset
    (https://labsites.rochester.edu/air/projects/URMP.html).
    """

    @staticmethod
    def available_splits():
        """
        Obtain a list of available (pre-defined) dataset splits.

        Returns
        ----------
        splits : list of strings
          Numbers which indicate each overarching piece in the dataset
        """

        splits = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10',
                  '11', '12', '13', '14', '15', '16', '17', '18', '19', '20',
                  '21', '22', '23', '24', '25', '26', '27', '28', '29', '30',
                  '31', '32', '33', '34', '35', '36', '37', '38', '39', '40',
                  '41', '42', '43', '44']

        return splits

    @classmethod
    def download(cls, save_dir):
        """
        Download the URMP dataset to a specified location.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of URMP
        """

        # Create top-level directory
        super().download(save_dir)

        # URL pointing to the tar file containing data for all tracks
        url = 'https://datadryad.org/stash/downloads/file_stream/99348'

        # Construct a path for saving the file
        tar_path = os.path.join(save_dir, 'URMP.tar.gz')

        # Download the tar file
        stream_url_resource(url, tar_path, 1000 * 1024)

        # Untar the downloaded file and remove it
        unzip_and_remove(tar_path, tar=True)

        # Move contents of unzipped directory to the base directory
        change_base_dir(save_dir, os.path.join(save_dir, 'Dataset'))
