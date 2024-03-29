# Baseline Model for Zero-Shot Voice Cloning

This document outlines the setup and usage instructions for the baseline model of the Zero-Shot Voice Cloning project.

## Dataset Preparation

1. Download the dataset from [Edinburgh DataShare](https://datashare.ed.ac.uk/handle/10283/3443).
2. Unzip the dataset into a folder named `VCTK-Corpus-0.92`. Ensure this folder contains the `txt` and `wav48_silence_trimmed` subfolders.

## Pretrained Models Preparation

1. Download `hubert_base.pt`, `D.pth`, and `G.pth` files from [this Google Drive link](https://drive.google.com/drive/folders/1AXG5Pj4oG2emvf9jxEkR-EbTmpyWLJVR?usp=sharing) to the `pretrained` folder within your project directory.
2. Download `added.index` and `p225.pth` files to `VCTK-Corpus-0.92/p225`.

## Environment Setup

The environment for this project is based on the Docker image from [PyTorch on DockerHub](https://hub.docker.com/r/pytorch/pytorch/tags). Please refer to the `requirements.txt` file for additional Python package requirements. Note that GCC (GNU Compiler Collection) is required for some dependencies.

## Inference

To run the inference:
1. Open and run the `infer.ipynb` Jupyter notebook.
2. Modify the `load_audio` function call to use your input audio file, e.g., `audio = load_audio('VCTK-Corpus-0.92/wav48_silence_trimmed/p228/p228_002_mic1.flac', 16000)`.
3. Set `person = 'VCTK-Corpus-0.92/p225/p225.pth'` and `file_index = 'VCTK-Corpus-0.92/p225/added.index'` to use the target voice identity for cloning.

## Training

For training:
1. Run `preprocess.py` with the following specifications to prepare your data:
   - `inp_root = 'VCTK-Corpus-0.92/wav48_silence_trimmed/p225'` (Input directory containing audio files for fine-tuning)
   - `exp_dir = 'VCTK-Corpus-0.92/p225'` (Output directory for processed data)
2. Then, execute `train.py` to start the training process:
   - `name = 'p225'` (Specify the model name)
   - `exp_dir = 'VCTK-Corpus-0.92/p225'` (Data source directory, should match the `exp_dir` from the preprocessing step)

Please ensure that you have followed the dataset and pretrained models preparation steps before proceeding with environment setup, inference, and training phases.

## Example
The `example` folder contains an example showcasing the model's input and output:

- `Input_Audio.flac`: The model's input audio file.
- `Model_Output.wav`: The output audio file generated by the model.
- `Target_tone_color_1.flac` and `Target_tone_color_2.flac`: Audio files of the voice to be cloned.