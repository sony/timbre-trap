from torch.utils.data import default_collate
from tqdm import tqdm

import requests
import tarfile
import zipfile
import shutil
import os


__all__ = [
    'constants',
    'stream_url_resource',
    'unzip_and_remove',
    'change_base_dir',
    'separate_ground_truth'
]


class constants(object):
    DEFAULT_LOCATION = os.path.join(os.path.expanduser('~'), 'Desktop', 'Datasets')
    KEY_TRACK = 'track'
    KEY_AUDIO = 'audio'
    KEY_TIMES = 'times'
    KEY_GROUND_TRUTH = 'ground-truth'


def stream_url_resource(url, save_path, chunk_size=1024):
    """
    Download a file at a URL by streaming it.

    Parameters
    ----------
    url : string
      URL pointing to the file
    save_path : string
      Path to the save location (including the file name)
    chunk_size : int
      Number of bytes to download at a time
    """

    print(f'Downloading {os.path.basename(url)}...')

    # Create an HTTP GET request
    r = requests.get(url,
                     stream=True,
                     headers={'Accept-Encoding': None})
    # Obtain iterable chunks of the resource
    chunks = r.iter_content(chunk_size)

    # Determine unit of one iteration
    unit_scale = chunk_size / 1000 / 1000

    # Determine total number of bytes to download
    total_length = r.headers.get('content-length')

    if total_length is not None:
        # Create a stream with a progress bar
        stream = tqdm(unit='MB',
                      unit_scale=unit_scale,
                      total=(int(total_length) / chunk_size),
                      bar_format='{l_bar}{bar}| {n:.1f}/{total:.1f}MB [{elapsed}<{remaining}, ''{rate_fmt}{postfix}]')
    else:
        # Create a stream without a progress bar
        stream = tqdm(unit='MB',
                      unit_scale=unit_scale,
                      bar_format='{n:.1f}MB [{elapsed}, ''{rate_fmt}{postfix}]')

    # Open the target file in write mode
    with open(save_path, 'wb') as file:
        # Download all chunks
        for chunk in chunks:
            # Check if chunk download was successful
            if chunk:
                # Write the chunk to the file
                file.write(chunk)
                # Manually update the progress bar
                stream.update(len(chunk) / chunk_size)
        stream.close()


def unzip_and_remove(zip_path, target=None, tar=False):
    """
    Unzip a zip file and remove it.

    Parameters
    ----------
    zip_path : string
      Path to the zip file
    target : string or None
      Directory to extract the zip file contents into
    tar : bool
      Whether the compressed file is in TAR format
    """

    print(f'Unzipping {os.path.basename(zip_path)}...')

    if target is None:
        # Default save location to same directory
        target = os.path.dirname(zip_path)

    if tar:
        # Open the tar file in read mode
        with tarfile.open(zip_path, 'r') as tar_ref:
            # Extract contents into target directory
            tar_ref.extractall(target)
    else:
        # Open the zip file in read mode
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            # Extract contents into target directory
            zip_ref.extractall(target)

    # Delete the zip file
    os.remove(zip_path)


def change_base_dir(new_dir, old_dir):
    """
    Change the top-level directory from the path chain of each file.

    Parameters
    ----------
    new_dir : string
      New top-level directory
    old_dir : string
      Old top-level directory
    """

    # Loop through all contents of old directory
    for content in os.listdir(old_dir):
        # Construct the old path to the contents
        old_path = os.path.join(old_dir, content)
        # Construct the new path to the contents
        new_path = os.path.join(new_dir, content)
        # Move the file or subdirectory
        shutil.move(old_path, new_path)

    # Remove (now empty) old top-level directory
    os.rmdir(old_dir)


def separate_ground_truth(batch):
    """
    Collate a batch into groups based on data availability.

    Parameters
    ----------
    batch : list of dicts containing
      track : string
        Identifier for the track
      ...

    Returns
    ----------
    data_both : None or dict containing
      tracks : list of string
        Identifiers for the batched tracks
      audio : Tensor (B x 1 x N)
        Sampled audio for batched tracks
      times : Tensor (B x T)
        Corresponding frame times for batched tracks
      ground_truth : Tensor (B x F x T)
        Ground-truth activations for batched tracks
    data_audio : None or dict containing
      tracks : list of string
        Identifiers for the batched tracks
      audio : Tensor (B x 1 x N)
        Sampled audio for batched tracks
    data_score : None or dict containing
      tracks : list of string
        Identifiers for the batched tracks
      times : Tensor (B x T)
        Corresponding frame times for batched tracks
      ground_truth : Tensor (B x F x T)
        Ground-truth activations for batched tracks
    """

    # Initialize lists for each type of data
    data_both = list()
    data_audio = list()
    data_score = list()

    while len(batch):
        # Check which data entries exist
        entries = list(batch[0].keys())

        if constants.KEY_AUDIO in entries and constants.KEY_GROUND_TRUTH in entries:
            # Add data to audio/ground-truth list
            data_both.append(batch.pop(0))
        elif constants.KEY_AUDIO in entries:
            # Add data to audio-only list
            data_audio.append(batch.pop(0))
        elif constants.KEY_GROUND_TRUTH in entries:
            # Add data to score-only list
            data_score.append(batch.pop(0))
        else:
            return NotImplementedError

    # Collate each group using standard procedure, defaulting to None if there is no data
    data_both = default_collate(data_both) if len(data_both) else None
    data_audio = default_collate(data_audio) if len(data_audio) else None
    data_score = default_collate(data_score) if len(data_score) else None

    return data_both, data_audio, data_score
