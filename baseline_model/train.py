import utils
from data_utils import TextAudioLoader, DistributedBucketSampler, TextAudioCollate
import os
from random import shuffle, randint
import pathlib
import json
from torch.utils.data import DataLoader
from models import SynthesizerTrn, MultiPeriodDiscriminator
import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from mel_processing import spec_to_mel_torch, mel_spectrogram_torch
import commons
from losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
)
from torch.nn import functional as F
from time import sleep
import torch.distributed as dist

def train_and_evaluate(
    rank, epoch, hps, nets, optims, schedulers, scaler, loaders, cache
):
    net_g, net_d = nets
    optim_g, optim_d = optims
    train_loader, eval_loader = loaders

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()

    # Prepare data iterator
    if hps.if_cache_data_in_gpu == True:
        # Use Cache
        data_iterator = cache
        if cache == []:
            # Make new cache
            for batch_idx, info in enumerate(train_loader):
                # Unpack
                (
                    phone,
                    phone_lengths,
                    spec,
                    spec_lengths,
                    wave,
                    wave_lengths,
                    sid,
                ) = info
                # Load on CUDA
                if torch.cuda.is_available():
                    phone = phone.cuda(rank, non_blocking=True)
                    phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
                    sid = sid.cuda(rank, non_blocking=True)
                    spec = spec.cuda(rank, non_blocking=True)
                    spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
                    wave = wave.cuda(rank, non_blocking=True)
                    wave_lengths = wave_lengths.cuda(rank, non_blocking=True)
                # Cache on list
                cache.append(
                    (
                        batch_idx,
                        (
                            phone,
                            phone_lengths,
                            spec,
                            spec_lengths,
                            wave,
                            wave_lengths,
                            sid,
                        ),
                    )
                )
        else:
            # Load shuffled cache
            shuffle(cache)
    else:
        # Loader
        data_iterator = enumerate(train_loader)

    for batch_idx, info in data_iterator:
        # Data
        ## Unpack
        phone, phone_lengths, spec, spec_lengths, wave, wave_lengths, sid = info
        ## Load on CUDA
        if (hps.if_cache_data_in_gpu == False) and torch.cuda.is_available():
            phone = phone.cuda(rank, non_blocking=True)
            phone_lengths = phone_lengths.cuda(rank, non_blocking=True)
            sid = sid.cuda(rank, non_blocking=True)
            spec = spec.cuda(rank, non_blocking=True)
            spec_lengths = spec_lengths.cuda(rank, non_blocking=True)
            wave = wave.cuda(rank, non_blocking=True)
            # wave_lengths = wave_lengths.cuda(rank, non_blocking=True)

        # Calculate
        with autocast(enabled=hps.train.fp16_run):
            (
                y_hat,
                ids_slice,
                x_mask,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = net_g(phone, phone_lengths, spec, spec_lengths, sid)
            mel = spec_to_mel_torch(
                spec,
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.mel_fmin,
                hps.data.mel_fmax,
            )
            y_mel = commons.slice_segments(
                mel, ids_slice, hps.train.segment_size // hps.data.hop_length
            )
            with autocast(enabled=False):
                y_hat_mel = mel_spectrogram_torch(
                    y_hat.float().squeeze(1),
                    hps.data.filter_length,
                    hps.data.n_mel_channels,
                    hps.data.sampling_rate,
                    hps.data.hop_length,
                    hps.data.win_length,
                    hps.data.mel_fmin,
                    hps.data.mel_fmax,
                )
            if hps.train.fp16_run == True:
                y_hat_mel = y_hat_mel.half()
            wave = commons.slice_segments(
                wave, ids_slice * hps.data.hop_length, hps.train.segment_size
            )  # slice

            # Discriminator
            y_d_hat_r, y_d_hat_g, _, _ = net_d(wave, y_hat.detach())
            with autocast(enabled=False):
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                    y_d_hat_r, y_d_hat_g
                )
        optim_d.zero_grad()
        scaler.scale(loss_disc).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(wave, y_hat)
            with autocast(enabled=False):
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl
                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank == 0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]["lr"]
                # Amor For Tensorboard display
                if loss_mel > 75:
                    loss_mel = 75
                if loss_kl > 9:
                    loss_kl = 9
                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                    }
                )

                scalar_dict.update(
                    {"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)}
                )
                scalar_dict.update(
                    {"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)}
                )
        global_step += 1
    # /Run steps

    if epoch % hps.save_every_epoch == 0 and rank == 0:
        if hps.if_latest == 0:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(global_step)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(global_step)),
            )
        else:
            utils.save_checkpoint(
                net_g,
                optim_g,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "G_{}.pth".format(2333333)),
            )
            utils.save_checkpoint(
                net_d,
                optim_d,
                hps.train.learning_rate,
                epoch,
                os.path.join(hps.model_dir, "D_{}.pth".format(2333333)),
            )
        if rank == 0 and hps.save_every_weights == "1":
            if hasattr(net_g, "module"):
                ckpt = net_g.module.state_dict()
            else:
                ckpt = net_g.state_dict()
        if hasattr(net_g, "module"):
            ckpt = net_g.module.state_dict()
        else:
            ckpt = net_g.state_dict()
        utils.savee(
            ckpt, hps.sample_rate, hps.name, epoch, hps
        )
        sleep(1)
        os._exit(2333333)


os.environ["MASTER_ADDR"] = "localhost"
os.environ["MASTER_PORT"] = str(randint(20000, 55555))
dist.init_process_group(
    backend="gloo", init_method="env://", world_size=1, rank=0
)

name = 'p225'
exp_dir = 'VCTK-Corpus-0.92/p225'

n_gpus = 1
rank = 0

config_path = 'pretrained/config.json'
with open(config_path, "r") as f:
    d = json.load(f)
config_save_path = os.path.join(exp_dir, "config.json")
if not pathlib.Path(config_save_path).exists():
    with open(config_save_path, "w", encoding="utf-8") as f:
        json.dump(
            d,
            f,
            ensure_ascii=False,
            indent=4,
            sort_keys=True,
        )
        f.write("\n")

hps = utils.get_hparams(name, exp_dir)
train_dataset = TextAudioLoader(hps.data.training_files, hps.data)

train_sampler = DistributedBucketSampler(
    train_dataset,
    hps.train.batch_size * n_gpus,
    [100, 200, 300, 400, 500, 600, 700, 800, 900],  # 16s
    num_replicas=n_gpus,
    rank=rank,
    shuffle=True
)
collate_fn = TextAudioCollate()
train_loader = DataLoader(
    train_dataset,
    num_workers=4,
    shuffle=False,
    pin_memory=True,
    collate_fn=collate_fn,
    batch_sampler=train_sampler,
    persistent_workers=True,
    prefetch_factor=8,
)

net_g = SynthesizerTrn(
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    **hps.model,
    is_half=hps.train.fp16_run,
)
net_g = net_g.cuda(rank)

net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm)
net_d = net_d.cuda(rank)

optim_g = torch.optim.AdamW(
    net_g.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)
optim_d = torch.optim.AdamW(
    net_d.parameters(),
    hps.train.learning_rate,
    betas=hps.train.betas,
    eps=hps.train.eps,
)

net_g = DDP(net_g, device_ids=[rank])
net_d = DDP(net_d, device_ids=[rank])

try: 
    _, _, _, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d
    )
    _, _, _, epoch_str = utils.load_checkpoint(
        utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g
    )
    global_step = (epoch_str - 1) * len(train_loader)
except: 
    epoch_str = 1
    global_step = 0
    if hps.pretrainG != "":
        if hasattr(net_g, "module"):
            net_g.module.load_state_dict(
                torch.load(hps.pretrainG, map_location="cpu")["model"]
            )
        else:
            net_g.load_state_dict(
                    torch.load(hps.pretrainG, map_location="cpu")["model"]
                )
    if hps.pretrainD != "":
        if hasattr(net_d, "module"):
            net_d.module.load_state_dict(
                torch.load(hps.pretrainD, map_location="cpu")["model"]
            )
        else:
            net_d.load_state_dict(
                torch.load(hps.pretrainD, map_location="cpu")["model"]
            )

scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
    optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
    optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str - 2
)

scaler = GradScaler(enabled=hps.train.fp16_run)

cache = []
for epoch in range(epoch_str, hps.train.epochs + 1):
    train_and_evaluate(
        rank,
        epoch,
        hps,
        [net_g, net_d],
        [optim_g, optim_d],
        [scheduler_g, scheduler_d],
        scaler,
        [train_loader, None],
        cache
    )
    scheduler_g.step()
    scheduler_d.step()