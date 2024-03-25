import torch
import torchaudio
from main.VAE_main import AudioVAE_Flow
from main.utils import extract_vec_feats, extract_tonecolor_feats, extract_text_features


def split_and_pad_tensor(source, split_size=4096):
    num_splits = (source.size(0) + split_size - 1) // split_size
    extended_source = torch.zeros(num_splits * split_size, dtype=source.dtype)
    extended_source[:source.size(0)] = source
    split_tensors = extended_source.split(split_size)
    return split_tensors


def extract_features(split_tensors):
    reference_vec_feature = []
    reference_tone_feature = []
    reference_text_feature = []
    for chunk in split_tensors:
        vec_feature = extract_vec_feats(chunk)
        tone_feature = extract_tonecolor_feats(chunk)
        text_feature = extract_text_features(chunk)
        vec_feature_normalized = (vec_feature - vec_feature.mean()) / vec_feature.std()
        tone_feature_normalized = (tone_feature - tone_feature.mean()) / tone_feature.std()
        text_feature_normalized = (text_feature - text_feature.mean()) / text_feature.std()
        reference_vec_feature.append(vec_feature_normalized)
        reference_tone_feature.append(tone_feature_normalized)
        reference_text_feature.append(text_feature_normalized)
    return reference_vec_feature, reference_tone_feature, reference_text_feature


def process_audio(source_path, reference_path, checkpoint_path, device):
    source, _ = torchaudio.load(source_path)
    reference, _ = torchaudio.load(reference_path)
    split_source_tensors = split_and_pad_tensor(source.squeeze(0))
    split_reference_tensors = split_and_pad_tensor(reference.squeeze(0))

    model = AudioVAE_Flow(device=device).to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['state_dict'])

    reference_vec_feature, reference_tone_feature, reference_text_feature = extract_features(split_reference_tensors)
    vec_feature_mean = torch.stack(reference_vec_feature).mean(dim=0)
    tone_feature_mean = torch.stack(reference_tone_feature).mean(dim=0)
    text_feature_mean = torch.stack(reference_text_feature).mean(dim=0)

    generated = []
    for source_chunk in split_source_tensors:
        mean = source_chunk.mean()
        std = source_chunk.std()
        source_chunk_normalized = (source_chunk - mean) / std
        source_vec_feature = extract_vec_feats(source_chunk_normalized)
        source_tone_feature = extract_tonecolor_feats(source_chunk_normalized)
        source_text_feature = extract_text_features(source_chunk_normalized)
        generated_audio = model(None, source_text_feature, source_tone_feature, source_vec_feature, text_feature_mean,
                                tone_feature_mean, vec_feature_mean, mode='valid')
        generated_audio_denormalized = generated_audio * std + mean
        generated.append(generated_audio_denormalized.squeeze(2))

    generated = torch.cat(generated, dim=1)  # [:source.squeeze(0).size(0)]
    # generated = generated.squeeze(0)
    torchaudio.save('output_audio.flac', generated.detach(), 48000)



if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    process_audio("./p225_001_mic1.flac", "./p226_001_mic1.flac",
                  "/Users/liuziwei/Desktop/GNN/version3.0 copyorg/main/model_checkpoint.pth.tar", device)
