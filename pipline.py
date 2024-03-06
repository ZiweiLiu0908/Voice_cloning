import logging
import os

import torch
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from DataLoader import CustomDataset
from Ecapa_TDNN import SpeakerToneColorExtractor
from model import loss_function, AudioVAE_RealNVP
from utils import load_checkpoint, save_checkpoint

# 检查是否有可用的GPU，并据此设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
logging.basicConfig(filename='org_training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')


def process():
    # 初始化模型和优化器
    model = AudioVAE_RealNVP().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 检查是否有保存的模型和训练状态
    checkpoint_path = "model_checkpoint.pth.tar"
    if os.path.isfile(checkpoint_path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        start_epoch = load_checkpoint(checkpoint, model, optimizer)
        print(f"=> Loaded checkpoint at epoch {start_epoch}")
    else:
        print("=> No checkpoint found, starting from scratch")

    #  dict_keys(['speaker_id', 'audio', 'file', 'text', 'text_id', 'age', 'gender', 'accent', 'region', 'comment'])
    ######################################################################################################################
    path = 'vctk'
    # 实例化 Feature Extractor
    feature_extractor = SpeakerToneColorExtractor()
    # 实例化自定义数据集
    custom_dataset = CustomDataset(dataset_path=path, feature_extractor=feature_extractor)
    custom_dataset = DataLoader(custom_dataset, batch_size=32, shuffle=True, num_workers=1)
    ######################################################################################################################

    # 训练模型
    num_epochs = 1000
    for epoch in tqdm(range(num_epochs)):
        for mel_spectrogram_db, source_embeddings, reference_embeddings in custom_dataset:
            mel_spectrogram_db = mel_spectrogram_db.to(device)
            source_embeddings = source_embeddings.to(device)
            reference_embeddings = reference_embeddings.to(device)
            # 前向传播
            generated_x, x, mu, logvar, z, z_inversed = model(mel_spectrogram_db, source_embeddings, reference_embeddings)
            # 计算损失
            loss, recon_loss, kl_loss, nvp_loss = loss_function(generated_x, x, mu, logvar, z, z_inversed)
            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # print(
        #     f"Epoch {epoch + 1}, Loss: {loss}, Recon: {recon_loss.item()}, KL: {kl_loss.item()}, NVP: {nvp_loss.item()}")
        logging.info(
            f"Epoch {epoch + 1}, Meta Loss: {loss.item()}, Recon Loss: {recon_loss.item()}, KL Loss: {kl_loss.item()}, NVP Loss: {nvp_loss.item()}")
        # 保存模型和训练状态
        if (epoch + 1) % 100 == 0:  # 每100个epoch保存一次
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })


if __name__ == '__main__':
    process()
