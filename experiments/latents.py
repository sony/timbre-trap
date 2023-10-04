from timbre_trap.datasets.MixedMultiPitch import Bach10 as Bach10_Mixtures
from timbre_trap.datasets.SoloMultiPitch import Bach10 as Bach10_Stems
from timbre_trap.datasets import constants
from utils import *

from torch.utils.data import DataLoader
from matplotlib import rcParams
from tqdm import tqdm

import shutil
import torch
import os


# Name of the model to evaluate
ex_name = 'Final_Base_V2'

# Choose the model checkpoint to compare
checkpoint = 8750

# Choose the GPU on which to perform evaluation
gpu_id = None

# Construct the path to the top-level directory of the experiment
experiment_dir = os.path.join('..', 'generated', 'experiments', ex_name)

# Random seed for evaluation
seed = 0


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050


###########
## MODEL ##
###########

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Construct the path to the model checkpoint to evaluate
model_path = os.path.join(experiment_dir, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the disentanglement model
model = torch.load(model_path, map_location=device)
model.eval()


##############
## DATASETS ##
##############

# Use the default base directory paths
bch10_base_dir = None

# Legend for labeling instrument within stems
bach10_instrument_legend = ['Violin', 'Clarinet', 'Saxophone', 'Bassoon']

# Obtain a list of tracks from original Bach10 dataset
mixture_tracks = Bach10_Mixtures(base_dir=bch10_base_dir,
                                 sample_rate=sample_rate,
                                 cqt=model.sliCQ).tracks

# Define counts for readability
num_tracks, num_stems = 10, 4


###################
## VISUALIZATION ##
###################

# Construct a path to the directory under which to save visualizations
save_dir = os.path.join(experiment_dir, 'visualization')

# Make sure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Seed everything with the same seed
seed_everything(seed)

# Create an empty list to hold all latents and instrument labels
all_latents, instruments = list(), list()

# Loop through each mixture
for i in tqdm(range(num_tracks)):
    print(f'Computing latents for stems from track \'{mixture_tracks[i]}\'...')

    # Instantiate all Bach10 dataset stems belonging to a mixture
    bch10_stems = Bach10_Stems(base_dir=bch10_base_dir,
                               splits=[f'{i + 1:02d}'],
                               sample_rate=sample_rate,
                               cqt=model.sliCQ)

    # Initialize a PyTorch dataloader for a batch of the stems
    loader = DataLoader(dataset=bch10_stems, batch_size=num_stems, shuffle=False)

    # Load a batch of stems
    for stems in loader:
        # Obtain track names
        track = stems[constants.KEY_TRACK]
        # Extract audio and add to the appropriate device
        audio = stems[constants.KEY_AUDIO].to(device)

        # Add fixed set of instruments for stems
        instruments += bach10_instrument_legend

        with torch.no_grad():
            # Pad audio to next multiple of block length
            audio = model.sliCQ.pad_to_block_length(audio)

            # Obtain latent vectors for each stem
            latents, _, _ = model.encode(audio)

            # Add latents to tracked list
            all_latents.append(latents.mean(-1))

# Concatenate latents from all stems
all_latents = torch.cat(all_latents, dim=0)

# Construct a path to save latent space visualization for checkpoint
save_path = os.path.join(save_dir, f'latents-{checkpoint}.pdf')

# Change the font and font size for the plot
rcParams['font.size'] = 20

# Visualize latents produced by the encoder by instrument
plot_latents(all_latents, instruments, save_path=save_path)
