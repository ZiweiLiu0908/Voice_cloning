import torch
from speechbrain.inference import SpeakerRecognition


class SpeakerToneColorExtractor:
    def __init__(self, model_source="speechbrain/spkrec-ecapa-voxceleb", savedir="tmpdir"):
        # 加载 Ecapa-TDNN 预训练模型
        self.model = SpeakerRecognition.from_hparams(source=model_source, savedir=savedir,  run_opts={'device': 'cuda'})

    def extract_features(self, audio):
        # 音频数据预处理：确保音频是单通道的，并且是浮点张量
        if not isinstance(audio, torch.FloatTensor):
            audio = torch.FloatTensor(audio)  # 转换为浮点张量
        if audio.dim() == 1:
            audio = audio.unsqueeze(0)  # 添加批次维度

        # audio = audio.to('cuda')
        # 使用 Ecapa-TDNN 模型提取特征'
        # print(audio.device)
        embeddings = self.model.encode_batch(audio).squeeze(0)

        return embeddings