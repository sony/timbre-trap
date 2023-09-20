from .. import BaseDataset


class Bach10(BaseDataset):
    """
    Implements the top-level wrapper for the Bach10 dataset
    (https://labsites.rochester.edu/air/resource.html).
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

        splits = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10']

        return splits

    @classmethod
    def download(cls, save_dir):
        """
        At this time, Bach10 must be downloaded manually so an error is thrown.

        Parameters
        ----------
        save_dir : string
          Directory under which to save the contents of Bach10
        """

        return NotImplementedError
