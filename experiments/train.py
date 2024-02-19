from timbre_trap.datasets.MixedMultiPitch import URMP as URMP_Mixtures, Bach10 as Bach10_Mixtures, Su, MusicNet as MusicNet_Mixtures, TRIOS
from timbre_trap.datasets.SoloMultiPitch import URMP as URMP_Stems, MusicNet as MusicNet_Stems, MedleyDB_Pitch, MAESTRO, GuitarSet
from timbre_trap.datasets.AudioStems import MedleyDB as MedleyDB_Stems
from timbre_trap.datasets.AudioMixtures import MedleyDB as MedleyDB_Mixtures, FMA
from timbre_trap.datasets import ComboDataset, StemMixingDataset
from timbre_trap.framework import TimbreTrap

from timbre_trap.framework.objectives import *
from timbre_trap.utils import *
from evaluate import evaluate

from torch.utils.tensorboard import SummaryWriter
from sacred.observers import FileStorageObserver
from torch.utils.data import DataLoader
from sacred import Experiment
from tqdm import tqdm

import warnings
import torch
import math
import os


EX_NAME = '_'.join(['Final_Base_V2_Demo'])

ex = Experiment('Train a model to reconstruct and transcribe audio')


@ex.config
def config():
    ##############################
    ## TRAINING HYPERPARAMETERS ##
    ##############################

    # Specify a checkpoint from which to resume training (None to disable)
    checkpoint_path = None

    # Maximum number of training iterations to conduct
    max_epochs = 5000

    # Number of iterations between checkpoints
    checkpoint_interval = 250

    # Number of samples to gather for a batch
    batch_size = 4

    # Number of seconds of audio per sample
    n_secs = 9

    # Initial learning rate
    learning_rate = 1e-3

    # Scaling factors for each loss term
    multipliers = {
        'reconstruction' : 1,
        'transcription' : 1,
        'consistency' : 1
    }

    # Number of epochs spanning warmup phase (0 to disable)
    n_epochs_warmup = 50

    # Set validation dataset to compare for learning rate decay and early stopping
    validation_criteria_set = URMP_Mixtures.name()

    # Set validation metric to compare for learning rate decay and early stopping
    validation_criteria_metric = 'mpe/f1-score'

    # Select whether the validation criteria should be maximized or minimized
    validation_criteria_maximize = True # (False - minimize | True - maximize)

    # Late starting point (0 to disable)
    n_epochs_late_start = 0

    # Number of epochs without improvement before reducing learning rate (0 to disable)
    n_epochs_decay = 500

    # Number of epochs before starting epoch counter for learning rate decay
    n_epochs_cooldown = 100

    # Number of epochs without improvement before early stopping (None to disable)
    n_epochs_early_stop = None

    # IDs of the GPUs to use, if available
    gpu_ids = [0]

    # Random seed for this experiment
    seed = 2

    ########################
    ## FEATURE EXTRACTION ##
    ########################

    # Number of samples per second of audio
    sample_rate = 22050

    # Number of octaves the CQT should span
    n_octaves = 9

    # Number of bins in a single octave
    bins_per_octave = 60

    ############
    ## OTHERS ##
    ############

    # Number of threads to use for data loading
    n_workers = 8 * len(gpu_ids)

    # Top-level directory under which to save all experiment files
    root_dir = os.path.join('..', 'generated', 'experiments', EX_NAME)

    # Create the root directory
    os.makedirs(root_dir, exist_ok=True)

    # Flag to make experimental setup more lightweight
    debug = False

    if debug:
        # Print a warning message indicating debug mode is active
        warnings.warn('Running in DEBUG mode...', RuntimeWarning)

    # Add a file storage observer for the log directory
    ex.observers.append(FileStorageObserver(root_dir))


@ex.automain
def train_model(checkpoint_path, max_epochs, checkpoint_interval, batch_size, n_secs, learning_rate, multipliers,
                n_epochs_warmup, validation_criteria_set, validation_criteria_metric, validation_criteria_maximize,
                n_epochs_late_start, n_epochs_decay, n_epochs_cooldown, n_epochs_early_stop, gpu_ids, seed,
                sample_rate, n_octaves, bins_per_octave, n_workers, root_dir, debug):
    # Discard read-only types
    multipliers = dict(multipliers)
    gpu_ids = list(gpu_ids)

    # Seed everything with the same seed
    seed_everything(seed)

    # Use the default base directory paths
    urmp_base_dir = None
    mydb_ptch_base_dir = None
    mydb_base_dir = None
    mstro_base_dir = None
    bch10_base_dir = None
    su_base_dir = None
    mnet_base_dir = None
    gset_base_dir = None
    fma_base_dir = None
    trios_base_dir = None

    # Initialize the primary PyTorch device
    device = torch.device(f'cuda:{gpu_ids[0]}'
                          if torch.cuda.is_available() else 'cpu')

    if checkpoint_path is None:
        # Initialize autoencoder model and train from scratch
        model = TimbreTrap(sample_rate=sample_rate,
                           n_octaves=n_octaves,
                           bins_per_octave=bins_per_octave,
                           secs_per_block=3,
                           latent_size=128,
                           model_complexity=2,
                           skip_connections=False)
    else:
        # Load a preexisting model and resume training
        model = torch.load(checkpoint_path, map_location=device)

    if len(gpu_ids) > 1:
        # Wrap model for multi-GPU usage
        model = DataParallel(model, device_ids=gpu_ids)

    # Add model to primary device
    model = model.to(device)

    # Initialize lists to hold training datasets
    mpe_train, audio_train = list(), list()

    # Set the URMP validation set as was defined in the MT3 paper
    urmp_val_splits = ['01', '02', '12', '13', '24', '25', '31', '38', '39']

    # Allocate remaining tracks to URMP training set
    urmp_train_splits = URMP_Mixtures.available_splits()

    for t in urmp_val_splits:
        # Remove validation track
        urmp_train_splits.remove(t)

    if debug:
        # Instantiate URMP dataset validation mixtures for training
        urmp_mixes_train = URMP_Mixtures(base_dir=urmp_base_dir,
                                         splits=urmp_val_splits,
                                         sample_rate=sample_rate,
                                         cqt=model.sliCQ,
                                         n_secs=n_secs,
                                         seed=seed)
        mpe_train.append(urmp_mixes_train)
    else:
        # Instantiate URMP dataset stems for training
        urmp_stems_train = URMP_Stems(base_dir=urmp_base_dir,
                                      splits=urmp_train_splits,
                                      sample_rate=sample_rate,
                                      cqt=model.sliCQ,
                                      n_secs=n_secs,
                                      seed=seed)
        mpe_train.append(urmp_stems_train)

        # Instantiate URMP dataset mixtures for training
        urmp_mixes_train = URMP_Mixtures(base_dir=urmp_base_dir,
                                         splits=urmp_train_splits,
                                         sample_rate=sample_rate,
                                         cqt=model.sliCQ,
                                         n_secs=n_secs,
                                         seed=seed)
        mpe_train.append(urmp_mixes_train)

        # Instantiate MAESTRO dataset for training
        mstro_train = MAESTRO(base_dir=mstro_base_dir,
                              splits=['train'],
                              sample_rate=sample_rate,
                              cqt=model.sliCQ,
                              n_secs=n_secs,
                              seed=seed)
        #mpe_train.append(mstro_train)

        # Instantiate MedleyDB-Pitch subset for training
        mydb_ptch_train = MedleyDB_Pitch(base_dir=mydb_ptch_base_dir,
                                         splits=None,
                                         sample_rate=sample_rate,
                                         cqt=model.sliCQ,
                                         n_secs=n_secs,
                                         seed=seed)
        mpe_train.append(mydb_ptch_train)

        # Instantiate MedleyDB (audio-only) stems for training
        mydb_stems_train = MedleyDB_Stems(base_dir=mydb_base_dir,
                                          splits=None,
                                          sample_rate=sample_rate,
                                          n_secs=n_secs,
                                          seed=seed)
        #audio_train.append(mydb_stems_train)

        # Instantiate MedleyDB (audio-only) mixtures for training
        mydb_mixes_train = MedleyDB_Mixtures(base_dir=mydb_base_dir,
                                             splits=None,
                                             sample_rate=sample_rate,
                                             n_secs=n_secs,
                                             seed=seed)
        #audio_train.append(mydb_mixes_train)

        # Instantiate GuitarSet dataset for training
        gset_train = GuitarSet(base_dir=gset_base_dir,
                               splits=['00', '01', '02', '03', '04'],
                               sample_rate=sample_rate,
                               cqt=model.sliCQ,
                               n_secs=n_secs,
                               seed=seed)
        mpe_train.append(gset_train)

        # Instantiate MusicNet dataset for training
        mnet_stems_train = MusicNet_Stems(base_dir=mnet_base_dir,
                                          splits=['train'],
                                          sample_rate=sample_rate,
                                          cqt=model.sliCQ,
                                          n_secs=n_secs,
                                          seed=seed)
        #mpe_train.append(mnet_stems_train)

        # Instantiate MusicNet dataset for training
        mnet_mixes_train = MusicNet_Mixtures(base_dir=mnet_base_dir,
                                             splits=['train'],
                                             sample_rate=sample_rate,
                                             cqt=model.sliCQ,
                                             n_secs=n_secs,
                                             seed=seed)
        #mpe_train.append(mnet_mixes_train)

        # Instantiate FMA dataset (audio-only) for training
        fma_train = FMA(base_dir=fma_base_dir,
                        splits=None,
                        sample_rate=sample_rate,
                        n_secs=n_secs,
                        seed=seed)
        #audio_train.append(fma_train)

        # MPE datasets with combinable stems
        mpe_stem_sets = [urmp_stems_train,
                         mydb_ptch_train,
                         gset_train,
                         mnet_stems_train,
                         mstro_train]

        # Instantiate random mixtures from MPE stems for training
        rand_mpe_train = StemMixingDataset(mpe_stem_sets,
                                           sum([len(d) // 2 for d in mpe_stem_sets]),
                                           n_min=2,
                                           n_max=3,
                                           seed=seed)
        #mpe_train.append(rand_mpe_train)

        # Audio-only datasets with combinable stems
        audio_stem_sets = [mydb_stems_train]

        # Instantiate random mixtures from audio-only stems for training
        rand_audio_train = StemMixingDataset(audio_stem_sets,
                                             sum([len(d) // 2 for d in audio_stem_sets]),
                                             n_min=2,
                                             n_max=5,
                                             seed=seed)
        #audio_train.append(rand_audio_train)

    # Combine MPE and audio datasets
    mpe_train = ComboDataset(mpe_train)
    audio_train = ComboDataset(audio_train)

    if len(audio_train):
        # Split batch size and workers across data type
        audio_batch_size, audio_workers = batch_size // 2, n_workers // 2
    else:
        # There is no audio data to load
        audio_batch_size, audio_workers = 0, 0

    # Allocate batch size and workers for MPE data
    mpe_batch_size = batch_size - audio_batch_size
    mpe_workers = n_workers - audio_workers

    # Initialize a PyTorch dataloader for MPE data
    mpe_loader = DataLoader(dataset=mpe_train,
                            batch_size=mpe_batch_size,
                            shuffle=True,
                            num_workers=mpe_workers,
                            pin_memory=True,
                            drop_last=True)

    if len(audio_train):
        # Initialize a PyTorch dataloader for audio data
        audio_loader = DataLoader(dataset=audio_train,
                                  batch_size=audio_batch_size,
                                  shuffle=True,
                                  num_workers=audio_workers,
                                  pin_memory=True,
                                  drop_last=True)
    else:
        # Replace dataloader with null list
        audio_loader = [None] * len(mpe_loader)

    # Instantiate TRIOS dataset for validation
    trios_val = TRIOS(base_dir=trios_base_dir,
                      splits=None,
                      sample_rate=sample_rate,
                      cqt=model.sliCQ,
                      seed=seed)

    # Instantiate URMP dataset mixtures for validation
    urmp_mixes_val = URMP_Mixtures(base_dir=urmp_base_dir,
                                   splits=urmp_val_splits,
                                   sample_rate=sample_rate,
                                   cqt=model.sliCQ,
                                   seed=seed)

    # Instantiate GuitarSet dataset for evaluation
    gset_test = GuitarSet(base_dir=gset_base_dir,
                          splits=['05'],
                          sample_rate=sample_rate,
                          cqt=model.sliCQ,
                          seed=seed)

    # Instantiate MedleyDB-Pitch subset for validation
    mydb_ptch_val = MedleyDB_Pitch(base_dir=mydb_ptch_base_dir,
                                   splits=None,
                                   sample_rate=sample_rate,
                                   cqt=model.sliCQ,
                                   seed=seed)

    # Instantiate Bach10 dataset mixtures for evaluation
    bch10_test = Bach10_Mixtures(base_dir=bch10_base_dir,
                                 splits=None,
                                 sample_rate=sample_rate,
                                 cqt=model.sliCQ,
                                 seed=seed)

    # Instantiate Su dataset for evaluation
    su_test = Su(base_dir=su_base_dir,
                 splits=None,
                 sample_rate=sample_rate,
                 cqt=model.sliCQ,
                 seed=seed)

    # Add all validation datasets to a list
    validation_sets = [urmp_mixes_val, trios_val, bch10_test, su_test, gset_test]

    # Add all evaluation datasets to a list
    evaluation_sets = [urmp_mixes_val, trios_val, bch10_test, su_test, gset_test]

    # Initialize an optimizer for the model parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    # Determine the amount of batches in one epoch
    epoch_steps = min(len(mpe_loader), len(audio_loader))

    # Compute number of validation checkpoints corresponding to learning rate decay cooldown and window
    n_checkpoints_cooldown = math.ceil(n_epochs_cooldown * epoch_steps / checkpoint_interval)
    n_checkpoints_decay = math.ceil(n_epochs_decay * epoch_steps / checkpoint_interval)

    if n_epochs_early_stop is not None:
        # Compute number of validation checkpoints corresponding to early stopping window
        n_checkpoints_early_stop = math.ceil(n_epochs_early_stop * epoch_steps / checkpoint_interval)
    else:
        # Early stopping is disabled
        n_checkpoints_early_stop = None

    # Warmup global learning rate over a fixed number of steps according to a cosine function
    warmup_scheduler = CosineWarmup(optimizer, n_steps=n_epochs_warmup * epoch_steps)

    # Decay global learning rate by a factor of 1/2 after validation performance has plateaued
    decay_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                 mode='max' if validation_criteria_maximize else 'min',
                                                                 factor=0.5,
                                                                 patience=n_checkpoints_decay,
                                                                 threshold=2E-3,
                                                                 cooldown=n_checkpoints_cooldown)

    # Enable anomaly detection to debug any NaNs
    torch.autograd.set_detect_anomaly(True)

    # Construct the path to the directory for saving models
    log_dir = os.path.join(root_dir, 'models')

    # Initialize a writer to log results
    writer = SummaryWriter(log_dir)

    # Number of batches that have been processed
    batch_count = 0

    # Keep track of the model with the best validation results
    best_model_checkpoint = None

    # Keep track of the best model's results for comparison
    best_results = None

    # Counter for number of checkpoints since previous best results
    n_checkpoints_elapsed = 0

    # Flag to indicate early stopping criteria has been met
    early_stop_criteria = False

    # Loop through epochs
    for i in range(max_epochs):
        # Loop through all available batches, separating by type of data
        for data_mpe, data_audio in tqdm(zip(mpe_loader, audio_loader), desc=f'Epoch {i + 1}'):
            # Increment the batch counter
            batch_count += 1

            if warmup_scheduler.is_active():
                # Step the learning rate warmup scheduler
                warmup_scheduler.step()

            # Extract MPE data and add to appropriate device
            audio = data_mpe[constants.KEY_AUDIO].to(device)
            ground_truth = data_mpe[constants.KEY_GROUND_TRUTH].to(device).float()

            if data_audio is not None:
                # Concatenate audio-only data to MPE data audio
                audio = torch.cat((audio, data_audio[constants.KEY_AUDIO].to(device)))

            # Log the current learning rate for this batch
            writer.add_scalar('train/loss/learning_rate', optimizer.param_groups[0]['lr'], batch_count)

            # Obtain spectral coefficients of audio
            coefficients = model.sliCQ(audio)

            with torch.autocast(device_type=f'cuda'):
                # Perform transcription and reconstruction simultaneously
                reconstruction, latents, transcription_coeffs, \
                transcription_rec, transcription_scr, losses = model(audio, multipliers['consistency'])

                # Extract magnitude of decoded coefficients and convert to activations
                transcription = torch.nn.functional.tanh(model.sliCQ.to_magnitude(transcription_coeffs))

                # Compute the reconstruction loss for the batch
                reconstruction_loss = compute_reconstruction_loss(reconstruction, coefficients)
                # Log the reconstruction loss for this batch
                writer.add_scalar('train/loss/reconstruction', reconstruction_loss.item(), batch_count)

                # Compute the transcription loss for the batch
                transcription_loss = compute_transcription_loss(transcription[:mpe_batch_size], ground_truth, True)
                # Log the transcription loss for this batch
                writer.add_scalar('train/loss/transcription', transcription_loss.item(), batch_count)

                # Compute the total loss for this batch
                total_loss = multipliers['reconstruction'] * reconstruction_loss

                if i >= n_epochs_late_start:
                    # Add transcription loss to the total loss
                    total_loss += multipliers['transcription'] * transcription_loss

                    if multipliers['consistency']:
                        # Compute the consistency losses for the portion of the batch with ground-truth
                        consistency_loss_sp, consistency_loss_sc = compute_consistency_loss(transcription_rec[:mpe_batch_size],
                                                                                            transcription_scr[:mpe_batch_size],
                                                                                            transcription_coeffs[:mpe_batch_size])

                        # Log the (spectral-) consistency loss for this batch
                        writer.add_scalar('train/loss/consistency/spectral', consistency_loss_sp.item(), batch_count)

                        # Log the (transcription-) consistency loss for this batch
                        writer.add_scalar('train/loss/consistency/score', consistency_loss_sc.item(), batch_count)

                        # Compute the consistency loss for the batch
                        consistency_loss = consistency_loss_sp + consistency_loss_sc

                        # Add combined consistency loss to the total loss
                        total_loss += multipliers['consistency'] * consistency_loss

                for key_loss, val_loss in losses.items():
                    # Log the model loss for this batch
                    writer.add_scalar(f'train/loss/{key_loss}', val_loss.item(), batch_count)
                    # Add the model loss to the total loss
                    total_loss += multipliers.get(key_loss, 1) * val_loss

                # Log the total loss for this batch
                writer.add_scalar('train/loss/total', total_loss.item(), batch_count)

                # Zero the accumulated gradients
                optimizer.zero_grad()
                # Compute gradients using total loss
                total_loss.backward()

                # Compute the average gradient norm across the encoder
                avg_norm_encoder = average_gradient_norms(model.encoder)
                # Log the average gradient norm of the encoder for this batch
                writer.add_scalar('train/avg_norm/encoder', avg_norm_encoder, batch_count)
                # Determine the maximum gradient norm across encoder
                max_norm_encoder = get_max_gradient_norm(model.encoder)
                # Log the maximum gradient norm of the encoder for this batch
                writer.add_scalar('train/max_norm/encoder', max_norm_encoder, batch_count)

                # Compute the average gradient norm across the decoder
                avg_norm_decoder = average_gradient_norms(model.decoder)
                # Log the average gradient norm of the decoder for this batch
                writer.add_scalar('train/avg_norm/decoder', avg_norm_decoder, batch_count)
                # Determine the maximum gradient norm across decoder
                max_norm_decoder = get_max_gradient_norm(model.decoder)
                # Log the maximum gradient norm of the decoder for this batch
                writer.add_scalar('train/max_norm/decoder', max_norm_decoder, batch_count)

                # Apply gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

                # Perform an optimization step
                optimizer.step()

            if batch_count % checkpoint_interval == 0:
                # Construct a path to save the model checkpoint
                model_path = os.path.join(log_dir, f'model-{batch_count}.pt')

                if isinstance(model, torch.nn.DataParallel):
                    # Unwrap and save the model
                    torch.save(model.module, model_path)
                else:
                    # Save the model as is
                    torch.save(model, model_path)

                # Initialize dictionary to hold all validation results
                validation_results = dict()

                for val_set in validation_sets:
                    # Validate the model checkpoint on each validation dataset
                    validation_results[val_set.name()] = evaluate(model=model,
                                                                  eval_set=val_set,
                                                                  multipliers=multipliers,
                                                                  writer=writer,
                                                                  i=batch_count,
                                                                  device=device)

                # Make sure model is on correct device and switch to training mode
                model = model.to(device)
                model.train()

                if decay_scheduler.patience and not warmup_scheduler.is_active() and i >= n_epochs_late_start:
                    # Step the learning rate decay scheduler by logging the validation metric for the checkpoint
                    decay_scheduler.step(validation_results[validation_criteria_set][validation_criteria_metric])

                # Extract the result on the specified metric from the validation results for comparison
                current_score = validation_results[validation_criteria_set][validation_criteria_metric]

                if best_results is not None:
                    # Extract the currently tracked best result on the specified metric for comparison
                    best_score = best_results[validation_criteria_set][validation_criteria_metric]

                if best_results is None or \
                        (validation_criteria_maximize and current_score > best_score) or \
                        (not validation_criteria_maximize and current_score < best_score):
                    # Set current checkpoint as best
                    best_model_checkpoint = batch_count
                    # Update best results
                    best_results = validation_results
                    # Reset number of checkpoints
                    n_checkpoints_elapsed = 0
                else:
                    # Increment number of checkpoints
                    n_checkpoints_elapsed += 1

                if n_checkpoints_early_stop is not None and n_checkpoints_elapsed >= n_checkpoints_early_stop:
                    # Early stop criteria has been reached
                    early_stop_criteria = True

                    break

        if early_stop_criteria:
            # Stop training
            break

    print(f'Achieved best results at {best_model_checkpoint} iterations...')

    for val_set in validation_sets:
        # Log the results at the best checkpoint for each validation dataset in metrics.json
        ex.log_scalar(f'Validation Results ({val_set.name()})', best_results[val_set.name()], best_model_checkpoint)

    # Construct a path to the best model checkpoint
    best_model_path = os.path.join(log_dir, f'model-{best_model_checkpoint}.pt')
    # Load the best model and make sure it is in evaluation mode
    best_model = torch.load(best_model_path, map_location=device)
    best_model.eval()

    for eval_set in evaluation_sets:
        # Evaluate the model using testing split
        final_results = evaluate(model=best_model,
                                 eval_set=eval_set,
                                 multipliers=multipliers,
                                 device=device)

        # Log the evaluation results for this dataset in metrics.json
        ex.log_scalar(f'Evaluation Results ({eval_set.name()})', final_results, best_model_checkpoint)
