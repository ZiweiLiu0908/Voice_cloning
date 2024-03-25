import os
import ffmpeg
import numpy as np
from scipy import signal
from utils import Slicer
from scipy.io import wavfile
import librosa
import fairseq
import torch
import soundfile as sf
import torch.nn.functional as F
from random import shuffle
from sklearn.cluster import MiniBatchKMeans
import faiss


def norm_write(tmp_audio, idx0, idx1):
        tmp_max = np.abs(tmp_audio).max()
        if tmp_max > 2.5:
            print("%s-%s-%s-filtered" % (idx0, idx1, tmp_max))
            return
        tmp_audio = (tmp_audio / tmp_max * (max * alpha)) + (
            1 - alpha
        ) * tmp_audio
        wavfile.write(
            "%s/%s_%s.wav" % (gt_wavs_dir, idx0, idx1),
            sr,
            tmp_audio.astype(np.float32),
        )
        tmp_audio = librosa.resample(
            tmp_audio, orig_sr=sr, target_sr=16000
        ) 
        wavfile.write(
            "%s/%s_%s.wav" % (wavs16k_dir, idx0, idx1),
            16000,
            tmp_audio.astype(np.float32),
        )

def load_audio(file, sr):
    # Load audio file using ffmpeg and resample as necessary
    out, _ = (
        ffmpeg.input(file, threads=0)
        .output("-", format="f32le", acodec="pcm_f32le", ac=1, ar=sr)
        .run(cmd=["ffmpeg", "-nostdin"], capture_stdout=True, capture_stderr=True)
    )
    return np.frombuffer(out, np.float32).flatten()

def pipeline(path, idx0):
    audio = load_audio(path, sr)
    audio = signal.lfilter(bh, ah, audio)
    idx1 = 0
    for audio in slicer.slice(audio):
        i = 0
        while 1:
            start = int(sr * (per - overlap) * i)
            i += 1
            if len(audio[start:]) > tail * sr:
                tmp_audio = audio[start : start + int(per * sr)]
                norm_write(tmp_audio, idx0, idx1)
                idx1 += 1
            else:
                tmp_audio = audio[start:]
                break
        norm_write(tmp_audio, idx0, idx1)

def pipeline_mp(infos):
    for path, idx0 in infos:
        pipeline(path, idx0)

def readwave(wav_path, normalize=False):
    wav, sr = sf.read(wav_path)
    assert sr == 16000
    feats = torch.from_numpy(wav).float()
    if feats.dim() == 2:  # double channels
        feats = feats.mean(-1)
    assert feats.dim() == 1, feats.dim()
    if normalize:
        with torch.no_grad():
            feats = F.layer_norm(feats, feats.shape)
    feats = feats.view(1, -1)
    return feats


sr = 40000
bh, ah = signal.butter(N=5, Wn=48, btype="high", fs=sr)
per = 3.7
inp_root = 'VCTK-Corpus-0.92/wav48_silence_trimmed/p225'
exp_dir = 'VCTK-Corpus-0.92/p225'
n_p = 10
overlap = 0.3
tail = per + overlap
max = 0.9
alpha = 0.75
spk_id = 0

gt_wavs_dir = "%s/gt_wavs" % exp_dir
wavs16k_dir = "%s/16k_wavs" % exp_dir
os.makedirs(exp_dir, exist_ok=True)
os.makedirs(gt_wavs_dir, exist_ok=True)
os.makedirs(wavs16k_dir, exist_ok=True)

slicer = Slicer(
            sr=sr,
            threshold=-42,
            min_length=1500,
            min_interval=400,
            hop_size=15,
            max_sil_kept=500,
        )

infos = [
            ("%s/%s" % (inp_root, name), idx)
            for idx, name in enumerate(sorted(list(os.listdir(inp_root))))
        ]

for i in range(n_p):
    pipeline_mp(infos[i::n_p])

is_half = True
feature_dir = (
    "%s/features" % exp_dir
)
os.makedirs(feature_dir, exist_ok=True)

device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "pretrained/hubert_base.pt"
models, saved_cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
    [model_path],
    suffix="",
)
model = models[0]
model = model.to(device)
if is_half:
    model = model.half()
model.eval()

todo = sorted(list(os.listdir(wavs16k_dir)))

for idx, file in enumerate(todo):
    if file.endswith(".wav"):
        wav_path = "%s/%s" % (wavs16k_dir, file)
        out_path = "%s/%s" % (feature_dir, file.replace("wav", "npy"))
        feats = readwave(wav_path, normalize=saved_cfg.task.normalize)
        padding_mask = torch.BoolTensor(feats.shape).fill_(False)
        inputs = {
            "source": (
                feats.half().to(device)
                if is_half and device not in ["mps", "cpu"]
                else feats.to(device)
            ),
            "padding_mask": padding_mask.to(device),
            "output_layer": 12
        }
        with torch.no_grad():
            logits = model.extract_features(**inputs)
            feats = (
                logits[0]
            )
        feats = feats.squeeze(0).float().cpu().numpy()
        if np.isnan(feats).sum() == 0:
            np.save(out_path, feats, allow_pickle=False)

names = set([name.split(".")[0] for name in os.listdir(gt_wavs_dir)]) & set(
    [name.split(".")[0] for name in os.listdir(feature_dir)]
)
opt = []
for name in names:
    opt.append(
        "%s/%s.wav|%s/%s.npy|%s"
        % (
            gt_wavs_dir.replace("\\", "\\\\"),
            name,
            feature_dir.replace("\\", "\\\\"),
            name,
            spk_id,
        )
    )
for _ in range(2):
    opt.append(
        "mute/mute.wav|mute/mute.npy|%s"
        % (spk_id)
    )
shuffle(opt)
with open("%s/filelist.txt" % exp_dir, "w") as f:
    f.write("\n".join(opt))



listdir_res = list(os.listdir(feature_dir))
npys = []

for name in sorted(listdir_res):
    phone = np.load("%s/%s" % (feature_dir, name))
    npys.append(phone)
big_npy = np.concatenate(npys, 0)
big_npy_idx = np.arange(big_npy.shape[0])
np.random.shuffle(big_npy_idx)
big_npy = big_npy[big_npy_idx]
if big_npy.shape[0] > 2e5:
    big_npy = (
        MiniBatchKMeans(
            n_clusters=10000,
            verbose=True,
            batch_size=256 * 1,
            compute_labels=False,
            init="random",
        )
        .fit(big_npy)
        .cluster_centers_
    )

n_ivf = min(int(16 * np.sqrt(big_npy.shape[0])), big_npy.shape[0] // 39)
index = faiss.index_factory(768, "IVF%s,Flat" % n_ivf)
index_ivf = faiss.extract_index_ivf(index)
index_ivf.nprobe = 1
index.train(big_npy)

batch_size_add = 8192
for i in range(0, big_npy.shape[0], batch_size_add):
    index.add(big_npy[i : i + batch_size_add])
faiss.write_index(
    index,
    "%s/added.index"
    % (exp_dir)
)
