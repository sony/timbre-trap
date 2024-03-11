## Instrument-Agnostic Multi-Pitch Estimation (Timbre-Trap)
Code for the paper "[Timbre-Trap: A Low-Resource Framework for Instrument-Agnostic Music Transcription](https://arxiv.org/abs/2309.15717)".
This repository contains the following (and more):
- Full Timbre-Trap framework
  - [NSGT-based invertible Constant-Q Transform (CQT)](https://github.com/archinetai/cqt-pytorch) wrapper
  - 2D autoencoder inspired by the [SoundStream](https://ieeexplore.ieee.org/abstract/document/9625818) architecture
  - Transcription, reconstruction, and consistency objectives
- Intuitive Multi-Pitch Estimation (MPE) and Automatic Music Transcription (AMT) dataset wrappers
- Training and evaluation scripts sufficient to reproduce experiments

Sonification and visualization demos are also available [here](https://sony.github.io/timbre-trap/).

## Installation
Clone the repository, install the requirements, then install ```timbre-trap```:
```
git clone https://github.com/sony/timbre-trap
pip install -r timbre-trap/requirements.txt
pip install -e timbre-trap/
```

## Usage
#### Dataset Wrappers
PyTorch datasets wrappers for several relevant datasets are available through the ```datasets``` subpackage.
Note that these are organized by data type, i.e. multi-instrument audio mixtures and single-instrument audio stems with and without accompanying annotations.
Some datasets have wrappers for both mixtures and stems.
The wrappers also differentiate between frame-level pitch (MPE) and note-level (AMT) annotations, depending on what is available for each dataset.


The following is an example of how to use a dataset wrapper:
```
from timbre_trap.datasets.MixedMultiPitch import URMP

urmp_data = URMP(base_dir=None,
                 splits=None,
                 sample_rate=22050,
                 cqt=None,
                 n_secs=None,
                 seed=0)

for track in urmp_data:
    name = data[constants.KEY_TRACK]
    audio = data[constants.KEY_AUDIO]

    times, multipitch = urmp_data.get_ground_truth(name)
```

By default, the wrapper will search for the top-level dataset directory at ```~/Desktop/Datasets/<DATASET>```.
However, the path can be specified using the ```base_dir``` keyword argument.
If the dataset does not exist, it will be downloaded automatically (except for in cases where this is not possible).

The ```splits``` keyword can be used to partition the data based on pre-defined attributes or metadata.
If overridden, it should be given some subset of the output from the ```available_splits()``` function for each respective wrapper.

A CQT module wrapper must be provided to the ```cqt``` argument in order to convert the ground-truth to targets during training:
```
from timbre_trap.framework import CQT

cqt_module = CQT(n_octaves=9,
                 bins_per_octave=60,
                 sample_rate=22050,
                 secs_per_block=3)

urmp_data.cqt = cqt_module

for track in urmp_data:
    name = data[constants.KEY_TRACK]
    audio = data[constants.KEY_AUDIO]

    ground_truth = data[constants.KEY_GROUND_TRUTH]
```

#### Using Timbre-Trap
The 2D autoencoder model used in the Timbre-Trap framework can be initialized with the following:
```
from timbre_trap.framework import Timbre-Trap

model = TimbreTrap(sample_rate=22050,
                   n_octaves=9,
                   bins_per_octave=60,
                   secs_per_block=3,
                   latent_size=128,
                   model_complexity=2,
                   skip_connections=False)
```

Under the hood this will also initialize a CQT module, accessible with ```model.cqt```, which provides several useful utilities.
These include conversion between real-valued and complex-valued coefficients, synthesis of audio coefficients, and acquisition of times for each frame of coefficients.

The script ```experiments/train.py``` exemplifies the training process for the framework.
It should be run with ```experiments/``` as the current working directory.
Relevant hyperparameters for experimentation are defined at the top of the script.

The training script also utilizes an evaluation loop defined in ```experiments/evaluate.py```, which can be invoked independently:
```
from evaluate import evaluate

results = evaluate(model=model,
                   eval_set=val_set,
                   multipliers=[1, 1, 1])
```

Helper functions are available for performing both reconstruction and transcription at inference time:
```
transcription = model.transcribe(audio)
reconstruction = model.reconstruct(audio)
```

The script ```experiments/comparison.py``` exemplifies using the Timbre-Trap framework for inference, and compares results to those computed for baseline models [Deep-Salience](https://github.com/rabitt/ismir2017-deepsalience) and [Basic-Pitch](https://github.com/spotify/basic-pitch).
Additional examples of inference are provided in the scripts ```experiments/latents.py``` and ```experiments/sonify.py```.
These scripts visualize a reduced latent space and synthesize audio from reconstructed spectral and transcription coefficients for Bach10, respectively.

## Generated Files
Execution of ```experiments/train.py``` will generate the following under ```<root_dir>``` (defined at the top of the script):
- ```n/``` - folder (beginning at ```n = 1```)<sup>1</sup> containing [sacred](https://sacred.readthedocs.io/en/stable/quickstart.html) experiment files:
  - ```config.json``` - parameter values used for the experiment
  - ```cout.txt``` - contains any text printed to console
  - ```metrics.json``` - validation and evaluation results for the best model checkpoints
  - ```run.json``` - system and experiment information
- ```models/``` - folder containing saved model weights at each checkpoint, as well as an events file (for each execution) readable by [tensorboard](https://www.tensorflow.org/tensorboard)
- ```_sources/``` - folder containing copies of scripts at the time(s) execution

<sup>1</sup>An additional folder (```n += 1```) containing similar files is created for each execution with the same experiment name ```<EX_NAME>```.

## Analysis
Losses and various validation metrics can be analyzed in real-time by running:
```
tensorboard --logdir=<root_dir>/models --port=<port>
```
Here we assume the current working directory contains ```<root_dir>```, and ```<port>``` is an integer corresponding to an available port (```port = 6006``` if unspecified).

After running the above command, navigate to [http://localhost:&lt;port&gt;]() with an internet browser to view any reported training or validation observations within the [tensorboard](https://www.tensorflow.org/tensorboard) interface.

## Cite
##### ICASSP 2024 Paper ([Link](https://arxiv.org/abs/2309.15717))
```
@article{cwitkowitz2024timbre,
  title     = {{Timbre-Trap}: A Low-Resource Framework for Instrument-Agnostic Music Transcription},
  author    = {Cwitkowitz, Frank and Cheuk, Kin Wai and Choi, Woosung and Mart{\'\i}nez-Ram{\'\i}rez, Marco A and Toyama, Keisuke and Liao, Wei-Hsiang and Mitsufuji, Yuki},
  year      = 2024,
  booktitle = {Proceedings of IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP)}
}
```
