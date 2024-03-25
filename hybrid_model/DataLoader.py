import random

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset

from utils import resample_audio, extract_tonecolor_feats, extract_text_features, extract_vec_feats

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class CustomDataset(Dataset):
    def __init__(self, dataset_path):
        self.dataset = load_from_disk(dataset_path)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        ref_idx = random.choice(range(len(self.dataset)))

        for ref in range(len(self.dataset)):
            if self.dataset[idx]['speaker_id'] == self.dataset[ref]['speaker_id']:
                continue
            if self.dataset[idx]['text_id'] == self.dataset[ref]['text_id']:
                ref_idx = ref
                break


        source_audio = self.dataset[idx]['audio']['array']
        target_audio = self.dataset[ref_idx]['audio']['array']

        source_audio = resample_audio(source_audio, target_length=4096)
        source_vec_feature = extract_vec_feats(source_audio)
        source_tone_feature = extract_tonecolor_feats(source_audio)
        source_text_feature = extract_text_features(source_audio)

        target_audio = resample_audio(target_audio, target_length=4096)
        reference_vec_feature = extract_vec_feats(target_audio)
        reference_tone_feature = extract_tonecolor_feats(target_audio)
        reference_text_feature = extract_text_features(target_audio)

        return target_audio, source_text_feature, source_tone_feature, source_vec_feature, \
            reference_text_feature, reference_tone_feature, reference_vec_feature
