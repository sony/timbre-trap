from timbre_trap.datasets.MixedMultiPitch import Bach10 as Bach10_Mixtures, Su
from timbre_trap.datasets.SoloMultiPitch import GuitarSet
from timbre_trap.datasets.NoteDataset import NoteDataset

from evaluate import MultipitchEvaluator
from timbre_trap.utils import *

from tqdm import tqdm

import numpy as np
import mir_eval
import librosa
import torch
import os


# Name of the model to evaluate
ex_name = '<EXPERIMENT_DIR>'

# Choose the model checkpoint to compare
checkpoint = 0

# Choose the GPU on which to perform evaluation
gpu_id = None

# Flag to print results for each track separately
verbose = True

# Construct the path to the top-level directory of the experiment
experiment_dir = os.path.join('..', 'generated', 'experiments', ex_name)


########################
## FEATURE EXTRACTION ##
########################

# Number of samples per second of audio
sample_rate = 22050


############
## MODELS ##
############

# Initialize a device pointer for loading the models
device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() and gpu_id is not None else 'cpu')

# Construct the path to the model checkpoint to evaluate
model_path = os.path.join(experiment_dir, 'models', f'model-{checkpoint}.pt')

# Load a checkpoint of the Timbre-Trap model
tt_mpe = torch.load(model_path, map_location=device)
tt_mpe.eval()


from basic_pitch.note_creation import model_frames_to_time
from basic_pitch import ICASSP_2022_MODEL_PATH
from basic_pitch.inference import predict
import tensorflow as tf

# Number of bins in a single octave
bp_bins_per_octave = 36
# Load the BasicPitch model checkpoint corresponding to paper
basic_pitch = tf.saved_model.load(str(ICASSP_2022_MODEL_PATH))
# Determine the MIDI frequency associated with each bin of Basic Pitch predictions
bp_midi_freqs = librosa.note_to_midi('A0') + np.arange(264) / (bp_bins_per_octave / 12)


# Specify the names of the files to download from GitHub
script_name, weights_name = 'predict_on_audio.py', 'multif0.h5'
# Construct a path to a top-level directory for DeepSalience
deep_salience_dir = os.path.join('..', 'generated', 'deep_salience')
# Create the necessary file hierarchy if it doesn't exist
os.makedirs(os.path.join(deep_salience_dir, 'weights'), exist_ok=True)
# Construct paths for the downloaded files
script_path = os.path.join(deep_salience_dir, script_name)
weights_path = os.path.join(deep_salience_dir, 'weights', weights_name)
try:
    # Attempt to import the DeepSalience inference code
    from generated.deep_salience.predict_on_audio import (model_def,
                                                          compute_hcqt,
                                                          get_single_test_prediction,
                                                          get_multif0)
except ModuleNotFoundError:
    # Point to the top-level directory containing files to download
    url_dir = 'https://raw.githubusercontent.com/rabitt/ismir2017-deepsalience/master/predict'
    # Construct the URLs of the files to download
    script_url, weights_url = f'{url_dir}/{script_name}', f'{url_dir}/weights/{weights_name}'
    # Download the script and weights files
    stream_url_resource(script_url, script_path)
    stream_url_resource(weights_url, weights_path)

    # Open the inference script for reading/writing
    with open(script_path, 'r+') as f:
        # Read all the code lines
        lines = f.readlines()  # Get a list of all lines
        # Reset file pointer
        f.seek(0)
        # Update lines of code
        lines[11] = 'from keras.layers import Input, Lambda, Conv2D, BatchNormalization\n'
        lines[69] = '\t\tBINS_PER_OCTAVE*N_OCTAVES, fmin=FMIN, bins_per_octave=BINS_PER_OCTAVE\n'
        # Remove outdated imports
        lines.pop(12)
        lines.pop(12)
        # Stop processing file
        f.truncate()
        # Overwrite the code
        f.writelines(lines)

    # Retry the original import
    from generated.deep_salience.predict_on_audio import (model_def,
                                                          compute_hcqt,
                                                          get_single_test_prediction,
                                                          get_multif0)
# Initialize DeepSalience
deep_salience = model_def()
# Load the weights from the paper
deep_salience.load_weights(weights_path)


##############
## DATASETS ##
##############

# Use the default base directory paths
bch10_base_dir = None
su_base_dir = None
gset_base_dir = None

# Instantiate Bach10 dataset mixtures for evaluation
bch10_test = Bach10_Mixtures(base_dir=bch10_base_dir,
                             splits=None,
                             sample_rate=sample_rate,
                             cqt=tt_mpe.sliCQ)

# Instantiate Su dataset for evaluation
su_test = Su(base_dir=su_base_dir,
             splits=None,
             sample_rate=sample_rate,
             cqt=tt_mpe.sliCQ)

# Instantiate GuitarSet dataset for evaluation
gset_test = GuitarSet(base_dir=gset_base_dir,
                      splits=['05'],
                      sample_rate=sample_rate,
                      cqt=tt_mpe.sliCQ)


################
## EVALUATION ##
################

# Construct a path to the directory under which to save comparisons
save_dir = os.path.join(experiment_dir, 'comparisons')

# Make sure the comparison directory exists
os.makedirs(save_dir, exist_ok=True)

# Construct a path to the file to save the comparison results
save_path = os.path.join(save_dir, f'checkpoint-{checkpoint}.txt')

if os.path.exists(save_path):
    # Reset the file if it already exists
    os.remove(save_path)

# Loop through validation and evaluation datasets
for eval_set in [bch10_test, su_test, gset_test]:
    # Initialize evaluators for each algorithm/model
    ln_evaluator = MultipitchEvaluator()
    lg_evaluator = MultipitchEvaluator()
    tt_evaluator = MultipitchEvaluator()
    bp_evaluator = MultipitchEvaluator()
    ds_evaluator = MultipitchEvaluator()

    print_and_log(f'Results for {eval_set.name()}:', save_path)

    # Frequencies associated with ground-truth
    gt_midi_freqs = eval_set.cqt.get_midi_freqs()

    # Determine valid frequency bins for multi-pitch estimation (based on mir_eval)
    valid_freqs = librosa.midi_to_hz(gt_midi_freqs) > mir_eval.multipitch.MAX_FREQ

    # Loop through all tracks in the test set
    for i, data in enumerate(tqdm(eval_set)):
        # Determine which track is being processed
        track = data[constants.KEY_TRACK]
        # Extract audio and add to the appropriate device
        audio = data[constants.KEY_AUDIO].to(device).unsqueeze(0)

        if isinstance(eval_set, NoteDataset):
            # Extract frame times of ground-truth targets as reference
            times_ref = data[constants.KEY_TIMES]
            # Obtain the ground-truth note annotations
            pitches, intervals = eval_set.get_ground_truth(track)
            # Convert note pitches to Hertz
            pitches = librosa.midi_to_hz(pitches)
            # Convert the note annotations to multi-pitch annotations
            multi_pitch_ref = eval_set.notes_to_multi_pitch(pitches, intervals, times_ref)
        else:
            # Obtain the ground-truth multi-pitch annotations
            times_ref, multi_pitch_ref = eval_set.get_ground_truth(track)

        if verbose:
            # Print a header for the individual track's results
            print_and_log(f'\tResults for track \'{track}\' ({eval_set.name()}):', save_path)

        # Pad audio to next multiple of block length
        audio = eval_set.cqt.pad_to_block_length(audio)
        # Determine the times associated with features
        times_est = eval_set.cqt.get_times(eval_set.cqt.get_expected_frames(audio.size(-1)))
        # Obtain spectral coefficients of audio
        coefficients = eval_set.cqt(audio)
        # Obtain the magnitude of the coefficients
        magnitude = eval_set.cqt.to_magnitude(coefficients).squeeze(0)
        # Convert magnitude to linear gain between 0 and 1
        features_lin = magnitude / magnitude.max()
        # Obtain spectral features in decibels
        features_log = eval_set.cqt.to_decibels(magnitude)


        # Peak-pick and threshold the linear-scaled magnitude
        ln_activations = threshold(filter_non_peaks(to_array(features_lin)), 0.3)
        # Remove activations for invalid frequencies
        ln_activations[valid_freqs] = 0
        # Convert the raw-feature activations to frame-level multi-pitch estimates
        ln_multi_pitch = eval_set.activations_to_multi_pitch(ln_activations, gt_midi_freqs)
        # Compute results for predictions from the linear-scaled CQT features
        ln_results = ln_evaluator.evaluate(times_est, ln_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        ln_evaluator.append_results(ln_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(lin-cqt): {ln_results}', save_path)


        # Peak-pick and threshold the log-scaled magnitude
        lg_activations = threshold(filter_non_peaks(to_array(features_log)), 0.8)
        # Remove activations for invalid frequencies
        lg_activations[valid_freqs] = 0
        # Convert the raw-feature activations to frame-level multi-pitch estimates
        lg_multi_pitch = eval_set.activations_to_multi_pitch(lg_activations, gt_midi_freqs)
        # Compute results for predictions from the log-scaled CQT features
        lg_results = lg_evaluator.evaluate(times_est, lg_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        lg_evaluator.append_results(lg_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(log-cqt): {lg_results}', save_path)


        # Transcribe the audio using the Timbre-Trap model
        tt_activations = to_array(tt_mpe.transcribe(audio).squeeze())
        # Peak-pick and threshold the Timbre-Trap activations
        tt_activations = threshold(filter_non_peaks(tt_activations), 0.5)
        # Remove activations for invalid frequencies
        tt_activations[valid_freqs] = 0
        # Convert the Timbre-Trap activations to frame-level multi-pitch estimates
        tt_multi_pitch = eval_set.activations_to_multi_pitch(tt_activations, gt_midi_freqs)
        # Compute results for predictions from the Timbre-Trap methodology
        tt_results = tt_evaluator.evaluate(times_est, tt_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        tt_evaluator.append_results(tt_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(tt-mpe): {tt_results}', save_path)


        # Obtain a path for the track's audio
        audio_path = eval_set.get_audio_path(track)
        # Obtain predictions from the BasicPitch model
        model_output, _, _ = predict(audio_path, basic_pitch)
        # Extract the pitch salience predictions
        bp_salience = model_output['contour'].T
        # Determine times associated with each frame of predictions
        bp_times = model_frames_to_time(bp_salience.shape[-1])
        # Apply peak-picking and thresholding on the raw salience
        bp_salience = threshold(filter_non_peaks(bp_salience), 0.27)
        # Convert the activations to frame-level multi-pitch estimates
        bp_multi_pitch = eval_set.activations_to_multi_pitch(bp_salience, bp_midi_freqs)
        # Compute results for BasicPitch predictions
        bp_results = bp_evaluator.evaluate(bp_times, bp_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        bp_evaluator.append_results(bp_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(bsc-ptc): {bp_results}', save_path)


        # Compute features for DeepSalience model
        hcqt, freq_grid, time_grid = compute_hcqt(audio_path)
        # Obtain predictions from the DeepSalience model
        ds_salience = get_single_test_prediction(deep_salience, hcqt)
        # Convert the activations to frame-level multi-pitch estimates
        ds_times, ds_multi_pitch = get_multif0(ds_salience, freq_grid, time_grid, thresh=0.3)
        # Compute results for DeepSalience predictions
        ds_results = ds_evaluator.evaluate(ds_times, ds_multi_pitch, times_ref, multi_pitch_ref)
        # Store results for this track
        ds_evaluator.append_results(ds_results)

        if verbose:
            # Print results for the individual track
            print_and_log(f'\t\t-(dp-slnc): {ds_results}', save_path)

    # Print a header for average results across all tracks of the dataset
    print_and_log(f'\tAverage Results ({eval_set.name()}):', save_path)

    # Print average results
    print_and_log(f'\t\t-(lin-cqt): {ln_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(log-cqt): {lg_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(tt-mpe): {tt_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(bsc-ptc): {bp_evaluator.average_results()[0]}', save_path)
    print_and_log(f'\t\t-(dp-slnc): {ds_evaluator.average_results()[0]}', save_path)
