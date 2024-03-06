import torch
from datasets import load_from_disk, load_dataset
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import librosa


class CustomDataset(Dataset):
    def __init__(self, dataset_path, feature_extractor):
        self.dataset = self.load_dataset(dataset_path)
        self.feature_extractor = feature_extractor

    def load_dataset(self, path):
        # 这里应该是加载数据集的代码，但因为我们无法直接执行它，所以假装它返回了一个数据集
        dataset = load_dataset(path, split='train')
        return dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # 从数据集中随机选择一个作为 reference audio
        ref_idx = random.choice(range(len(self.dataset)))
        while self.dataset[ref_idx]['speaker_id'] == self.dataset[idx]['speaker_id']:  # 确保 source 和 reference 不是同一个
            ref_idx = random.choice(range(len(self.dataset)))

        # 提取 source 和 reference 的音频数组
        source_audio = self.dataset[idx]['audio']['array']
        reference_audio = self.dataset[ref_idx]['audio']['array']

        # 将 source_audio 转换为 NumPy 数组
        source_audio_np = np.array(source_audio)

        # 将 source_audio 转换为梅尔频谱
        mel_spec = librosa.feature.melspectrogram(y=source_audio_np, sr=48000, n_mels=128, fmax=8000)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        target_shape = (128, 44)
        # 首先确保梅尔频率维度正确（这应该已经通过 n_mels=128 参数设置）
        assert mel_spec_db.shape[0] == target_shape[0], "梅尔频率带数量不匹配"

        mel_spec_db_resized = librosa.util.fix_length(mel_spec_db, size=target_shape[1], axis=1)
        mel_spectrogram_db_tensor = torch.from_numpy(mel_spec_db_resized).float()
        mel_spectrogram_db_tensor = mel_spectrogram_db_tensor.unsqueeze(0)

        # 使用 SpeakerFeatureExtractor 提取 embedding vectors
        source_embedding = self.feature_extractor.extract_features(source_audio_np)
        reference_embedding = self.feature_extractor.extract_features(np.array(reference_audio))

        return mel_spectrogram_db_tensor, source_embedding, reference_embedding
