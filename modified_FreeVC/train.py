import logging
import os
import json
import argparse
import itertools
import math
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

import commons
import utils
from data_utils import (
  TextAudioSpeakerLoader,
  TextAudioSpeakerCollate,
  DistributedBucketSampler
)
from models import (
  SynthesizerTrn,
  MultiPeriodDiscriminator,
)
from losses import (
  generator_loss,
  discriminator_loss,
  feature_loss,
  kl_loss
)
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

class StepLR:
    def __init__(self, optimizer,total_steps=0, step_size=25, gamma=0.1):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.total_steps = total_steps

    def step(self):
        # Increment the total steps count
        self.total_steps += 1
        
        if self.total_steps % self.step_size == 0:
            # Update learning rate
            for param_group in self.optimizer.param_groups:
                if param_group['lr'] > 5e-6:
                  param_group['lr'] *= self.gamma
  
        # print("Learning rate: ", self.optimizer.param_groups[0]['lr'])

torch.backends.cudnn.benchmark = True
global_step = 0
# os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO


def main():
  """Assume Single Node Multi GPUs Training Only"""
  assert torch.cuda.is_available(), "CPU training is not allowed."
  hps = utils.get_hparams()

  n_gpus = torch.cuda.device_count()
  os.environ['MASTER_ADDR'] = 'localhost'
  os.environ['MASTER_PORT'] = hps.train.port

  # mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))
  run(0, 1, hps)


def run(rank, n_gpus, hps):
  global global_step
  if rank == 0:
    logger = utils.get_logger(hps.model_dir)
    logger.info(hps)
    utils.check_git_hash(hps.model_dir)
    writer = SummaryWriter(log_dir=hps.model_dir)
    writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

  # dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
  torch.manual_seed(hps.train.seed)
  torch.cuda.set_device(rank)

  train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps) # load training dataset
  train_sampler = DistributedBucketSampler(
      train_dataset,
      hps.train.batch_size,
      [32,300,400,500,600,700,800,900,1000],
      num_replicas=n_gpus,
      rank=rank,
      shuffle=True)
  collate_fn = TextAudioSpeakerCollate(hps)
  train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
      collate_fn=collate_fn, batch_sampler=train_sampler)
  if rank == 0:
    eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps)
    eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=True,
        batch_size=hps.train.batch_size, pin_memory=False,
        drop_last=False, collate_fn=collate_fn)

  net_g = SynthesizerTrn(
      hps.data.filter_length // 2 + 1,
      hps.train.segment_size // hps.data.hop_length,
      **hps.model).cuda(rank)
  net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

  # net_g = DDP(net_g, device_ids=[rank])#, find_unused_parameters=True)
  # net_d = DDP(net_d, device_ids=[rank])

  try:
    # Create new optimizer to load from checkpoint
    optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
    optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
    
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path("./checkpoints/training", "G_*.pth"), net_g, optim_g) # Load optimizer as well
    _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path("./checkpoints/training", "D_*.pth"), net_d, optim_d)
    global_step = (epoch_str - 1) * len(train_loader)
  except:
    # Loading pretrained Generator model
    utils.load_checkpoint("checkpoints/freevc.pth", net_g, None, True) # Do not load optimizer  
    # Loading pretrained Discriminator model
    utils.load_checkpoint("checkpoints/D-freevc.pth", net_d, None, True)
  
    # Create new optimizer, only loads parameters from the checkpoint
    optim_g = torch.optim.AdamW(
      net_g.parameters(), 
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps) 
    optim_d = torch.optim.AdamW(
      net_d.parameters(),
      hps.train.learning_rate, 
      betas=hps.train.betas, 
      eps=hps.train.eps)
    
    epoch_str = 1
    global_step = 0

  # # Use old scheduler for training
  # scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  # scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
  # Use new scheduler for training
  scheduler_g = StepLR(optim_g, step_size=25, gamma=hps.train.lr_decay)
  scheduler_d = StepLR(optim_d, step_size=25, gamma=hps.train.lr_decay)

  scaler = GradScaler(enabled=hps.train.fp16_run)

  for epoch in range(epoch_str, hps.train.epochs + 1):
    if rank==0:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
    else:
      train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)



def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
  
  net_g, net_d = nets
  optim_g, optim_d = optims
  scheduler_g, scheduler_d = schedulers
  train_loader, eval_loader = loaders
  if writers is not None:
    writer, writer_eval = writers

  train_loader.batch_sampler.set_epoch(epoch)
  global global_step

  net_g.train()
  net_d.train()
  for batch_idx, items in tqdm(enumerate(train_loader)):
    if hps.model.use_spk:
      c, spec, y, spk = items
      g = spk.cuda(rank, non_blocking=True)
    else:
      c, spec, y = items
      g = None
    spec, y = spec.cuda(rank, non_blocking=True), y.cuda(rank, non_blocking=True)
    c = c.cuda(rank, non_blocking=True)
    mel = spec_to_mel_torch(
          spec, 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate,
          hps.data.mel_fmin, 
          hps.data.mel_fmax) # convert spectrom to mel spectrogram

    with autocast(enabled=hps.train.fp16_run):
      # Generator
      y_hat, ids_slice, z_mask,\
      (_, z_p, m_p, logs_p, _ , _) = net_g(c, spec, g=g, mel=mel, multi_samples = True) # forward, get all outputs before decoder
      #(z, z_p, m_p, logs_p, m_q, logs_q)
      # c: audio, spec: spectrogram, g: speaker embedding, mel: mel spectrogram
      # z_p: high-dim normal distribution, should only contain content information
      # m_p: prior mean, logs_p: prior log of standard deviation
      # m_q: posterior mean, logs_q: posterior log of standard deviation
      # print("Got net_g: z_p, logs_q, m_p, logs_p, z_mask")

      y_mel = commons.slice_segments(mel, ids_slice, hps.train.segment_size // hps.data.hop_length)
      y_hat_mel = mel_spectrogram_torch(
          y_hat.squeeze(1), 
          hps.data.filter_length, 
          hps.data.n_mel_channels, 
          hps.data.sampling_rate, 
          hps.data.hop_length, 
          hps.data.win_length, 
          hps.data.mel_fmin, 
          hps.data.mel_fmax
      )
      y = commons.slice_segments(y, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

      # Discriminator
      y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())
      # print("Got net_d: y_d_hat_r, y_d_hat_g")
      with autocast(enabled=False):
        loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g)
        loss_disc_all = loss_disc
    optim_d.zero_grad()
    scaler.scale(loss_disc_all).backward()
    scaler.unscale_(optim_d)
    grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
    scaler.step(optim_d)
    # print("Updated optim_d")

    with autocast(enabled=hps.train.fp16_run):
      # Discriminator
      y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat)
      # print("Got net_d: y_d_hat_r, y_d_hat_g, fmap_r, fmap_g")
      with autocast(enabled=False):
        # F1 distance between real and generated mel-spectrogram, multiplied by a factor
        loss_mel = F.l1_loss(y_mel, y_hat_mel) * hps.train.c_mel
        # print("Got loss_mel")

        # KL divergence between prior and posterior, measure the difference bewteen z' and mu_p, logs_p
        # prior: mean = m_p, log variance = logs_p
        # posterior: mean = z_p, log variance = logs_q
        # z_p: high-dim normal distribution, should only contain content information
        # m_p: prior mean, logs_p: prior log of standard deviation
        # m_q: posterior mean, logs_q: posterior log of standard deviation
        loss_kl = kl_loss(z_p, m_p, logs_p, z_mask) * hps.train.c_kl
        # print("Got loss_kl")

        loss_fm = feature_loss(fmap_r, fmap_g)
        # print("Got loss_fm")
        loss_gen, losses_gen = generator_loss(y_d_hat_g)
        # print("Got loss_gen")
        # add loss here
        loss_gen_all = loss_gen + loss_fm + loss_mel + loss_kl
    optim_g.zero_grad()
    scaler.scale(loss_gen_all).backward()
    scaler.unscale_(optim_g)
    grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
    scaler.step(optim_g)
    scaler.update()
    # print("Updated optim_g")

    if rank==0:
      if global_step % hps.train.log_interval == 0:
        lr = optim_g.param_groups[0]['lr']
        losses = [loss_disc, loss_gen, loss_fm, loss_mel, loss_kl]
        logger.info('Train Epoch: {} [{:.0f}%]'.format(
          epoch,
          100. * batch_idx / len(train_loader)))
        logger.info([x.item() for x in losses] + [global_step, lr])
        
        scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm_d": grad_norm_d, "grad_norm_g": grad_norm_g}
        scalar_dict.update({"loss/g/fm": loss_fm, "loss/g/mel": loss_mel, "loss/g/kl": loss_kl})

        scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
        scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
        scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
        # add loss record here
        image_dict = { 
            "slice/mel_org": utils.plot_spectrogram_to_numpy(y_mel[0].data.cpu().numpy()),
            "slice/mel_gen": utils.plot_spectrogram_to_numpy(y_hat_mel[0].data.cpu().numpy()), 
            "all/mel": utils.plot_spectrogram_to_numpy(mel[0].data.cpu().numpy()),
        }
        utils.summarize(
          writer=writer,
          global_step=global_step, 
          images=image_dict,
          scalars=scalar_dict)

      if global_step % hps.train.eval_interval == 0:
        evaluate(hps, net_g, eval_loader, writer_eval)
        if not os.path.exists("./checkpoints/training"):
          os.makedirs("./checkpoints/training")
        utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join("./checkpoints/training", "G_{}.pth".format(global_step)))
        utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join("./checkpoints/training", "D_{}.pth".format(global_step)))
    
    scheduler_g.step()
    scheduler_d.step()
    global_step += 1
  
  if rank == 0:
    logger.info('====> Epoch: {}'.format(epoch))


 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    with torch.no_grad():
      for batch_idx, items in enumerate(eval_loader):
        if hps.model.use_spk:
          c, spec, y, spk = items
          g = spk[:1].cuda(0)
        else:
          c, spec, y = items
          g = None
        spec, y = spec[:1].cuda(0), y[:1].cuda(0)
        c = c[:1].cuda(0)
        break
      mel = spec_to_mel_torch(
        spec, 
        hps.data.filter_length, 
        hps.data.n_mel_channels, 
        hps.data.sampling_rate,
        hps.data.mel_fmin, 
        hps.data.mel_fmax)
      y_hat = generator.infer(c, g=g, mel=mel)
      
      y_hat_mel = mel_spectrogram_torch(
        y_hat.squeeze(1).float(),
        hps.data.filter_length,
        hps.data.n_mel_channels,
        hps.data.sampling_rate,
        hps.data.hop_length,
        hps.data.win_length,
        hps.data.mel_fmin,
        hps.data.mel_fmax
      )
    image_dict = {
      "gen/mel": utils.plot_spectrogram_to_numpy(y_hat_mel[0].cpu().numpy()),
      "gt/mel": utils.plot_spectrogram_to_numpy(mel[0].cpu().numpy())
    }
    audio_dict = {
      "gen/audio": y_hat[0],
      "gt/audio": y[0]
    }
    utils.summarize(
      writer=writer_eval,
      global_step=global_step, 
      images=image_dict,
      audios=audio_dict,
      audio_sampling_rate=hps.data.sampling_rate
    )
    generator.train()

                           
if __name__ == "__main__":
  main()
