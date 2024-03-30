from timbre_trap.datasets.MixedMultiPitch import Bach10 as Bach10_Mixtures
from timbre_trap.datasets.SoloMultiPitch import Bach10 as Bach10_Stems
from timbre_trap.utils import *

from torch.utils.data import DataLoader
from tqdm import tqdm

import soundfile as sf
import shutil
import torch
import os


# Name of the model to analyze
ex_name = '<EXPERIMENT_DIR>'

# Choose the model checkpoint to compare
checkpoint = 0

# Choose the GPU on which to perform sonification
gpu_id = None

# Construct the path to the top-level directory of the experiment
experiment_dir = os.path.join('..', 'generated', 'experiments', ex_name)


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050


###########
## MODEL ##
###########

# Initialize the chosen device
device = torch.device(f'cuda:{gpu_id}' if gpu_id is not None
                      and torch.cuda.is_available() else 'cpu')

# Construct the path to the model checkpoint to evaluate
model_path = os.path.join(experiment_dir, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the Timbre-Trap model
model = torch.load(model_path, map_location=device)
model.eval()


##############
## DATASETS ##
##############

# Use the default base directory paths
bch10_base_dir = None

# Instantiate Bach10 dataset mixtures for sonfication
bch10_mixes = Bach10_Mixtures(base_dir=bch10_base_dir,
                              splits=None,
                              sample_rate=sample_rate,
                              cqt=model.sliCQ)

# Legend for labeling instrument within stems
bach10_instrument_legend = ['Violin', 'Clarinet', 'Saxophone', 'Bassoon']


############
## SONIFY ##
############

# Construct a path to the directory under which to save synthesized audio
save_dir = os.path.join(experiment_dir, 'sonification', f'model-{checkpoint}')

if os.path.exists(save_dir):
    # Reset the directory if it already exists
    shutil.rmtree(save_dir)

# Make sure the save directory exists
os.makedirs(save_dir, exist_ok=True)

# Loop through mixtures
for i, data in enumerate(tqdm(bch10_mixes)):
    # Determine which mix is being processed
    track = data[constants.KEY_TRACK]
    # Extract mixture audio and add to the appropriate device
    audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)

    with torch.no_grad():
        # Pad audio to next multiple of block length
        audio = model.sliCQ.pad_to_block_length(audio)

        # Encode audio into coefficients and decode directly
        audio_ref = model.sliCQ.decode(model.sliCQ.encode(audio))

        # Encode raw audio into latent vectors
        latents, embeddings, losses = model.encode(audio)

        # Apply skip connections if they are turned on
        embeddings = model.apply_skip_connections(embeddings)

        # Decode latent vectors into spectral coefficients
        reconstruction = model.decode(latents, embeddings)

        # Estimate pitch using transcription switch
        transcription = model.decode(latents, embeddings, True)

        # Sonify reconstruction coefficients
        audio_rec = model.sliCQ.decode(reconstruction)

        # Sonify transcription coefficients
        audio_scr = model.sliCQ.decode(transcription)

    # Construct save paths for reference and reconstructed audio
    save_path_ref = os.path.join(save_dir, f'{track}_Mix_ref.wav')
    save_path_rec = os.path.join(save_dir, f'{track}_Mix_rec.wav')
    save_path_scr = os.path.join(save_dir, f'{track}_Mix_scr.wav')

    # Write reference and reconstructed audio to specified location
    sf.write(save_path_ref, to_array(audio_ref.squeeze()), sample_rate)
    sf.write(save_path_rec, to_array(audio_rec.squeeze()), sample_rate)
    sf.write(save_path_scr, to_array(audio_scr.squeeze()), sample_rate)

    # Instantiate all Bach10 dataset stems belonging to mixture
    bch10_stems = Bach10_Stems(base_dir=bch10_base_dir,
                               splits=[f'{i + 1:02d}'],
                               sample_rate=sample_rate,
                               cqt=model.sliCQ)

    # Initialize a PyTorch dataloader for a batch of the stems
    loader = DataLoader(dataset=bch10_stems, batch_size=len(bach10_instrument_legend), shuffle=False)

    # Load stems
    for stem in loader:
        # Extract stem audio and add to the appropriate device
        audio = stem[constants.KEY_AUDIO].to(device)

        with torch.no_grad():
            # Pad audio to next multiple of block length
            audio = model.sliCQ.pad_to_block_length(audio)

            # Encode audio into coefficients and decode directly
            audio_ref = model.sliCQ.decode(model.sliCQ.encode(audio))

            # Encode raw audio into latent vectors
            latents, embeddings, losses = model.encode(audio)

            # Apply skip connections if they are turned on
            embeddings = model.apply_skip_connections(embeddings)

            # Decode latent vectors into spectral coefficients
            reconstruction = model.decode(latents, embeddings)

            # Estimate pitch using transcription switch
            transcription = model.decode(latents, embeddings, True)

            # Sonify reconstruction coefficients
            audio_rec = model.sliCQ.decode(reconstruction)

            # Sonify transcription coefficients
            audio_scr = model.sliCQ.decode(transcription)

        for k, instrument in enumerate(bach10_instrument_legend):
            # Construct save paths for reference and reconstructed audio
            save_path_ref = os.path.join(save_dir, f'{track}_{instrument}_ref.wav')
            save_path_rec = os.path.join(save_dir, f'{track}_{instrument}_rec.wav')
            save_path_scr = os.path.join(save_dir, f'{track}_{instrument}_scr.wav')

            # Write reference and reconstructed audio to specified location
            sf.write(save_path_ref, to_array(audio_ref[k].squeeze()), sample_rate)
            sf.write(save_path_rec, to_array(audio_rec[k].squeeze()), sample_rate)
            sf.write(save_path_scr, to_array(audio_scr[k].squeeze()), sample_rate)
