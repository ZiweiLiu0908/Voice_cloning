**Installation Guide**
======================

### Download

From the Hybrid Model folder in https://drive.google.com/drive/folders/1wx9WVda4O5Fmfpx0zcNI0VrGtV9kHz5o?usp=sharing, download the model file **model_checkpoint.pth.tar** to the root directory of this folder. Download the model files **textFeat_model_checkpoint.pth.tar** and **toneFeat_model_checkpoint.pth.tar** and place them in the /extractor directory.



### Install the environment:

```shell
conda env export -f environment.yaml
```



### Training Process:

Place the training dataset (VCTK dataset was used in this project) in the root directory, then run:

```shell
python pipeline.py
```

The model will start training, and specific training parameters can be adjusted within pipeline.py.



### Inference Process:

Prepare the source audio and reference audio files and place them in the same directory as inference.py. Change the file names within inference.py.

```shell
python inference.py
```

This will generate the output file.