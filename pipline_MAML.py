import os
import higher
import torch
from torch import optim
from tqdm import tqdm

from DataLoader import CustomDataset
from Ecapa_TDNN import SpeakerToneColorExtractor
from model import loss_function, AudioVAE_RealNVP
from utils import load_checkpoint, save_checkpoint, split_task_data
from torch.utils.data import DataLoader
import logging

# 配置日志
logging.basicConfig(filename='MAML_training.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

device = 'cuda' if torch.cuda.is_available() else 'cpu'


def process():
    ######################################################################################################################
    path = 'vctk'
    # 实例化 Feature Extractor
    feature_extractor = SpeakerToneColorExtractor()
    # 实例化自定义数据集
    custom_dataset = CustomDataset(dataset_path=path, feature_extractor=feature_extractor)
    custom_dataset = DataLoader(custom_dataset, batch_size=32, shuffle=True, num_workers=1)
    ######################################################################################################################
    # 初始化模型和优化器
    model = AudioVAE_RealNVP(input_shape=(128, 44), latent_dim=20)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    # 修改优化器，以适应外循环更新
    meta_optimizer = optim.Adam(model.parameters(), lr=0.001)

    start_epoch = 0  # 默认从0开始
    num_epochs = 1000
    inner_steps = 5  # 每个任务的梯度更新步数

    # 检查是否有保存的模型和训练状态
    checkpoint_path = "model_checkpoint.pth.tar"
    if os.path.isfile(checkpoint_path):
        print("=> Loading checkpoint")
        checkpoint = torch.load(checkpoint_path)
        start_epoch = load_checkpoint(checkpoint, model, optimizer)
        print(f"=> Loaded checkpoint at epoch {start_epoch}")
    else:
        print("=> No checkpoint found, starting from scratch")

    # 开始训练
    for epoch in tqdm(range(start_epoch, num_epochs)):
        meta_optimizer.zero_grad()  # 清空外循环梯度
        print(111)

        for task_data in custom_dataset:  # 这里假设每个batch是一个新任务
            inputs_train, inputs_test = split_task_data(task_data)
            # 创建一个临时版本的模型，这个版本可以进行内循环更新而不影响原始模型
            with higher.innerloop_ctx(model, optimizer, copy_initial_weights=False) as (fmodel, diffopt):

                for _ in range(inner_steps):  # 对于每个任务执行多步内循环更新
                    # 假设 task_data 包含当前任务的输入和目标
                    mel_spectrogram_db, source_embeddings, reference_embeddings = inputs_train
                    generated_x, x, mu, logvar, z, z_inversed = fmodel(mel_spectrogram_db, source_embeddings,
                                                                       reference_embeddings)
                    loss, recon_loss, kl_loss, nvp_lossv = loss_function(generated_x, x, mu, logvar, z, z_inversed)
                    # 在内循环中更新模型
                    diffopt.step(loss)

                mel_spectrogram_db, source_embeddings, reference_embeddings = inputs_test
                # 在内循环结束后，计算对原始模型参数的梯度并进行累加
                generated_x, x, mu, logvar, z, z_inversed = fmodel(mel_spectrogram_db, source_embeddings,
                                                                   reference_embeddings)
                meta_loss, recon_loss, kl_loss, nvp_lossv = loss_function(generated_x, x, mu, logvar, z,
                                                                          z_inversed)  # 计算元学习损失
                meta_loss.backward()  # 计算梯度

        # 执行外循环优化步骤
        meta_optimizer.step()

        # 打印损失信息等（根据需要调整）
        # print(f"Epoch {epoch + 1}, Meta Loss: {meta_loss.item()}")
        logging.info(
            f"Epoch {epoch + 1}, Meta Loss: {meta_loss.item()}, Recon Loss: {recon_loss.item()}, KL Loss: {kl_loss.item()}, NVP Loss: {nvp_lossv.item()}")
        # 根据需要保存模型状态
        if (epoch + 1) % 100 == 0:
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            })


if __name__ == '__main__':
    process()
