import logging
import os

import librosa
import numpy as np
import torch
from Ecapa_TDNN import SpeakerToneColorExtractor
from model import AudioVAE_RealNVP
from utils import denormalize_tensor_0_1

source_flac_file_path = './p225_001_mic1.flac'
reference_flac_file_path = './p226_001_mic1.flac'
# 读取FLAC文件
source_audio, sr1 = librosa.load(source_flac_file_path, sr=None)
reference_audio, sr2 = librosa.load(reference_flac_file_path, sr=None)

source_audio_np = np.array(source_audio)
reference_audio_np = np.array(reference_audio)

# 计算梅尔频谱
S = librosa.feature.melspectrogram(source_audio_np, sr=sr1, n_mels=128, fmax=8000)
target_shape = (128, 44)
# 转换为对数刻度
mel_spec_db = librosa.power_to_db(S, ref=np.max)
mel_spec_db_resized = librosa.util.fix_length(mel_spec_db, size=target_shape[1], axis=1)
mel_spectrogram_db_tensor = torch.from_numpy(mel_spec_db_resized).float()
mel_spectrogram_db_tensor = mel_spectrogram_db_tensor.unsqueeze(0)

feature_extractor = SpeakerToneColorExtractor()
source_embedding = feature_extractor.extract_features(source_audio_np)
reference_embedding = feature_extractor.extract_features(reference_audio_np)

model = AudioVAE_RealNVP()
generated_audio = model(mel_spectrogram_db_tensor, source_embedding, reference_embedding, mode='valid')
generated_audio_int16 = (generated_audio * 32767).short()  # 假设generated_audio是[-1,1]范围内的

from scipy.io.wavfile import write

# 设置采样率
sample_rate = 22050  # 或者任何适用于你数据的采样率

# 保存音频文件
write('generated_audio.wav', sample_rate, generated_audio_int16.numpy())
