from math import floor

import torchaudio
from torchaudio.transforms import InverseMelScale, GriffinLim
import torch.nn.functional as F
import torch


def db_to_amplitude(mel_spectrogram_db, ref=1.0):
    """
    将分贝单位的梅尔频谱转换回幅度形式。
    """
    amplitude = ref * torch.pow(torch.tensor(10.0), mel_spectrogram_db / 20.0)
    return amplitude


def mel_to_waveform(mel_spectrogram_db, sample_rate=22050, n_fft=2048, n_mels=128, win_length=2048, hop_length=512):
    """
    使用InverseMelScale和GriffinLim从梅尔频谱重建波形。
    """
    # 将dB梅尔频谱转换回功率梅尔频谱
    mel_spectrogram = db_to_amplitude(mel_spectrogram_db)

    # 使用InverseMelScale从梅尔频谱转换回线性频谱
    inverse_mel_scale = InverseMelScale(n_stft=n_fft // 2 + 1, n_mels=n_mels, sample_rate=sample_rate)
    linear_spectrogram = inverse_mel_scale(mel_spectrogram)

    # 使用GriffinLim算法从线性频谱重建波形
    griffin_lim = GriffinLim(n_fft=n_fft, win_length=win_length, hop_length=hop_length)
    waveform = griffin_lim(linear_spectrogram)

    return waveform


def save_reconstructed_mp3(reconstructed_mel):
    # 假设reconstructed_mel是重建的梅尔频谱
    waveform = mel_to_waveform(reconstructed_mel.squeeze(0))  # 移除批次维度
    # 确保波形形状正确
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)  # 添加通道维度

    # 尝试再次保存
    torchaudio.save("reconstructed_audio.wav", waveform, sample_rate=22050)


def load_audio_to_melspectrogram(file_path, n_mels=128):
    waveform, sample_rate = torchaudio.load(file_path)
    mel_spectrogram = torchaudio.transforms.MelSpectrogram(sample_rate, n_mels=n_mels)(waveform)
    mel_spectrogram_db = torchaudio.transforms.AmplitudeToDB()(mel_spectrogram)
    return mel_spectrogram_db


def adjust_melspectrogram_size(mel_spectrogram, target_shape):
    """
    调整梅尔频谱的尺寸以匹配目标形状，通过裁剪或填充实现。

    参数:
    mel_spectrogram (Tensor): 原始梅尔频谱张量，维度应该是 [channel, n_mels, time_steps]。
    target_shape (tuple): 目标形状 (n_mels, time_steps)。

    返回:
    Tensor: 调整尺寸后的梅尔频谱。
    """
    # 获取原始和目标的尺寸
    original_shape = mel_spectrogram.shape[-2:]
    delta_freq_bins = target_shape[0] - original_shape[0]
    delta_time_steps = target_shape[1] - original_shape[1]

    # 频率维度的调整
    if delta_freq_bins > 0:
        # 如果目标频率bins数大于原始频率bins数，进行填充
        padding = (0, 0, 0, delta_freq_bins)  # (左, 右, 下, 上)
        mel_spectrogram = F.pad(mel_spectrogram, padding, "constant", 0)
    elif delta_freq_bins < 0:
        # 如果目标频率bins数小于原始频率bins数，进行裁剪
        mel_spectrogram = mel_spectrogram[:, :target_shape[0], :]

    # 时间维度的调整
    if delta_time_steps > 0:
        # 如果目标时间步数大于原始时间步数，进行填充
        padding = (0, delta_time_steps, 0, 0)  # (左, 右, 下, 上)
        mel_spectrogram = F.pad(mel_spectrogram, padding, "constant", 0)
    elif delta_time_steps < 0:
        # 如果目标时间步数小于原始时间步数，进行裁剪
        mel_spectrogram = mel_spectrogram[:, :, :target_shape[1]]

    return mel_spectrogram




def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']


def read_audio_to_mel_spectrogram(path):
    mel_spectrogram_db = load_audio_to_melspectrogram(path)
    adjusted_mel_spectrogram = adjust_melspectrogram_size(mel_spectrogram_db, target_shape=(128, 44))
    assert adjusted_mel_spectrogram.shape[-2:] == (128, 44), "调整后的梅尔频谱尺寸与模型输入尺寸不匹配"
    return adjusted_mel_spectrogram.unsqueeze(0)  # 增加批次维度


def get_embedding_from_audio(path):
    extractor = SpeakerFeatureExtractor()
    embeddings = extractor.process_audio_file(path).unsqueeze(0)
    return embeddings


def normalize_tensor_0_1(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    normalized_tensor = (tensor - min_val) / (max_val - min_val)
    return normalized_tensor


def denormalize_tensor_0_1(normalized_tensor, min_val, max_val):
    tensor = normalized_tensor * (max_val - min_val) + min_val
    return tensor


def split_task_data(task_data, split_size=0.8):
    split_size = floor(task_data[0].size(0)*split_size)
    # 初始化两个列表来保存分割后的数据
    data_part_1 = []
    data_part_2 = []

    # 遍历task_data中的每个张量
    for tensor in task_data:
        # 将每个张量分割为两个部分，前12个样本和后4个样本
        part_1 = tensor[:split_size].squeeze(1)  # 提取前12个样本
        part_2 = tensor[split_size:].squeeze(1)  # 提取后4个样本

        # 将分割后的张量添加到对应的列表中
        data_part_1.append(part_1)
        data_part_2.append(part_2)

    return data_part_1, data_part_2
