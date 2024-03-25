import logging
import os
import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloader import CustomDataset
from main.Discriminator import Discriminator
from main.VAE_main import AudioVAE_Flow
from main.loss import loss_function
from utils import load_checkpoint, pre_process, save_checkpoint


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
logging.basicConfig(filename='training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def process(num_epochs, lr_step_size=500, lr_gamma=0.9):
    path = './vctk'
    custom_dataset = CustomDataset(dataset_path=path)
    model = AudioVAE_Flow(device=device).to(device)

    params_to_optimize = []
    for name, param in model.named_parameters():
        if 'tone_extractor' not in name and 'text_extractor' not in name:
            params_to_optimize.append(param)


    optimizer = optim.Adam(params=params_to_optimize, lr=0.00001)
    discriminator = Discriminator().to(device)
    disc_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    checkpoint_path = "model_checkpoint.pth.tar"
    if os.path.isfile(checkpoint_path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = load_checkpoint(checkpoint, model, optimizer)
        print(f"=> Loaded checkpoint at epoch {start_epoch}")
    else:
        print("=> No checkpoint found, starting from scratch")

    min_loss = float('inf')

    for epoch in range(num_epochs):
        for target_audio, source_text_feature, source_tone_feature, source_vec_feature, \
                reference_text_feature, reference_tone_feature, reference_vec_feature in custom_dataset:

            target_audio, source_text_feature, source_tone_feature, source_vec_feature, \
                reference_text_feature, reference_tone_feature, reference_vec_feature = pre_process(target_audio,
                                                                                                    source_text_feature,
                                                                                                    source_tone_feature,
                                                                                                    source_vec_feature,
                                                                                                    reference_text_feature,
                                                                                                    reference_tone_feature,
                                                                                                    reference_vec_feature,
                                                                                                    device=device)

            z_q, z_p, z_q_head, z_p_head, z_q_mu, z_q_log_var, z_p_mu, z_p_log_var, generated_audio = model(
                target_audio, source_text_feature, source_tone_feature, source_vec_feature,
                reference_text_feature, reference_tone_feature, reference_vec_feature)


            disc_optimizer.zero_grad()
            real_output = discriminator(target_audio)
            fake_output = discriminator(generated_audio.detach())
            disc_loss = -torch.mean(real_output) + torch.mean(fake_output)
            disc_loss.backward()
            disc_optimizer.step()


            total_loss, reconstruction_loss, kl_divergence_loss, z_p_head_loss, z_q_head_loss = loss_function(
                target_audio, z_q, z_p, z_q_head, z_p_head, z_q_mu, z_q_log_var, z_p_mu, z_p_log_var, generated_audio)


            fake_output = discriminator(generated_audio)
            gen_loss = -torch.mean(fake_output)
            total_loss += gen_loss

            if total_loss.item() < min_loss:
                min_loss = total_loss.item()
                save_checkpoint({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                })
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            logging.info(f'Epoch [{epoch + 1}/{num_epochs}], Total Loss: {total_loss.item():.4f}, '
                         f'Reconstruction Loss: {reconstruction_loss.item():.4f}, '
                         f'KL Divergence Loss: {kl_divergence_loss.item():.4f}, '
                         f'z_p_head Loss: {z_p_head_loss.item():.4f}, '
                         f'z_q_head Loss: {z_q_head_loss.item():.4f}, '
                         f'Generator Loss: {gen_loss.item():.4f}, '
                         f'Discriminator Loss: {disc_loss.item():.4f}')


if __name__ == '__main__':
    process(num_epochs=10)
