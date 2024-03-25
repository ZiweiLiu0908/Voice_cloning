import numpy as np
import torch
from speechbrain.inference import SpeakerRecognition
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector


def resample_audio(audio, target_length=4096):
    # resampled_audio = librosa.resample(audio, orig_sr=original_sr, target_sr=target_sr)
    audio = torch.from_numpy(audio).float()

    original_length = len(audio)


    if original_length < target_length:
        repeat_times = target_length // original_length + 1
        audio_repeated = np.tile(audio, repeat_times)[:target_length]


    elif original_length > target_length:
        start_point = np.random.randint(0, original_length - target_length)
        audio_repeated = audio[start_point:start_point + target_length]
    else:
        audio_repeated = audio

    return audio_repeated


feature_extractor_wavlm = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-plus-sv')
model_wavlm = WavLMForXVector.from_pretrained('microsoft/wavlm-base-plus-sv',
                                              output_loading_info=False)  # Suppress unused weights warning


def extract_vec_feats(audio_tensor, sampling_rate=16000):
    def repeat_tensor_until_length(tensor, target_length):
        repeat_times = target_length // tensor.size(0) + 1
        repeated_tensor = tensor.repeat(repeat_times)[:target_length]
        return repeated_tensor

    assert audio_tensor.dim() == 1 and audio_tensor.size(0) >= 4096


    min_length = 10000

    if audio_tensor.size(0) < min_length:
        audio_tensor = repeat_tensor_until_length(audio_tensor, min_length)

    inputs = feature_extractor_wavlm(audio_tensor, return_tensors="pt", sampling_rate=sampling_rate, padding="longest")
    embeddings = model_wavlm(**inputs).embeddings
    embeddings = torch.nn.functional.normalize(embeddings, dim=-1).cpu()
    return embeddings


class SpeakerToneColorExtractor:
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb", savedir="tools/tmpdir"):
        self.model = SpeakerRecognition.from_hparams(source=model_source, savedir=savedir)

    def extract_features(self, audio):

        if not isinstance(audio, torch.FloatTensor):
            audio = torch.FloatTensor(audio)
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)

        embeddings = self.model.encode_batch(audio).squeeze(0)

        return embeddings


def extract_tonecolor_feats(audio_tensor):
    model = SpeakerToneColorExtractor()
    return model.extract_features(audio_tensor)


processor_whisper = AutoProcessor.from_pretrained("openai/whisper-large-v3")
model_whisper = AutoModelForSpeechSeq2Seq.from_pretrained("openai/whisper-large-v3")


def extract_text_features(audio_tensor):
    """
    Extract features from an audio tensor using the OpenAI Whisper model.

    Args:
        audio_tensor (torch.Tensor): The input audio tensor.

    Returns:
        torch.Tensor: The extracted feature vector.
    """
    # Load model and processor

    # Preprocess audio
    input_features = processor_whisper(audio_tensor, sampling_rate=16000, return_tensors="pt").input_features

    # Ensure model is in eval mode
    model_whisper.eval()

    # Disable gradient computation
    with torch.no_grad():
        # Directly use the encoder of the model
        encoder_outputs = model_whisper.model.encoder(input_features=input_features)
        feature_vector = encoder_outputs.last_hidden_state.mean(dim=1)

    return feature_vector


def cosine_similarity(tensor_1, tensor_2):
    dot_product = torch.sum(tensor_1 * tensor_2, dim=1)
    norm_1 = torch.norm(tensor_1, dim=1)
    norm_2 = torch.norm(tensor_2, dim=1)
    cos_sim = dot_product / (norm_1 * norm_2 + 1e-8)
    return cos_sim


def pre_process(target_audio, source_text_feature, source_tone_feature, source_vec_feature,
                reference_text_feature, reference_tone_feature, reference_vec_feature, device='cuda:0'):

    target_audio = target_audio.to(device)
    source_text_feature = source_text_feature.to(device)
    source_tone_feature = source_tone_feature.to(device)
    source_vec_feature = source_vec_feature.to(device)
    reference_text_feature = reference_text_feature.to(device)
    reference_tone_feature = reference_tone_feature.to(device)
    reference_vec_feature = reference_vec_feature.to(device)


    source_text_feature = (source_text_feature - source_text_feature.mean()) / source_text_feature.std()
    source_tone_feature = (source_tone_feature - source_tone_feature.mean()) / source_tone_feature.std()
    source_vec_feature = (source_vec_feature - source_vec_feature.mean()) / source_vec_feature.std()
    reference_text_feature = (reference_text_feature - reference_text_feature.mean()) / reference_text_feature.std()
    reference_tone_feature = (reference_tone_feature - reference_tone_feature.mean()) / reference_tone_feature.std()
    reference_vec_feature = (reference_vec_feature - reference_vec_feature.mean()) / reference_vec_feature.std()


    target_audio = target_audio / torch.max(torch.abs(target_audio))
    target_audio = target_audio.unsqueeze(0).unsqueeze(1)

    return target_audio, source_text_feature, source_tone_feature, source_vec_feature, \
        reference_text_feature, reference_tone_feature, reference_vec_feature


def load_checkpoint(checkpoint, model, optimizer):
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    return checkpoint['epoch']



def save_checkpoint(state, filename="model_checkpoint.pth.tar"):
    torch.save(state, filename)