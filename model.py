import torch
import torch.nn as nn

import torch.nn.functional as F

from utils import normalize_tensor_0_1, denormalize_tensor_0_1


# 定义AffineCouplingLayer类
class AffineCouplingLayer(nn.Module):
    # 定义初始化方法
    def __init__(self, input_dim, condition_dim, hidden_dim=256):
        super(AffineCouplingLayer, self).__init__()
        self.input_dim = input_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim

        # 定义scale_net和translate_net网络
        self.scale_net = nn.Sequential(
            nn.Linear(self.input_dim // 2 + self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim // 2),
            nn.Tanh()
        )
        self.translate_net = nn.Sequential(
            nn.Linear(self.input_dim // 2 + self.condition_dim, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.input_dim // 2)
        )

    # 定义forward方法
    def forward(self, x, condition):
        # 将输入x分成两部分
        x1, x2 = torch.chunk(x, 2, dim=1)

        # 将条件向量和x的一部分合并用于网络输入
        condition_input = torch.cat((x1, condition), dim=1)

        # 计算尺度和平移参数
        scale = self.scale_net(condition_input)
        translate = self.translate_net(condition_input)

        # 应用仿射变换
        y2 = x2 * torch.exp(scale) + translate
        y = torch.cat((x1, y2), dim=1)

        return y

    # 定义inverse方法
    def inverse(self, y, condition):
        # 逆向过程分割y
        y1, y2 = torch.chunk(y, 2, dim=1)

        # 合并条件向量和y的一部分作为网络输入
        condition_input = torch.cat((y1, condition), dim=1)

        # 计算尺度和平移参数
        scale = self.scale_net(condition_input)
        translate = self.translate_net(condition_input)

        # 应用逆仿射变换
        x2 = (y2 - translate) * torch.exp(-scale)
        x = torch.cat((y1, x2), dim=1)

        return x


# 定义ConditionalRealNVP类
class ConditionalRealNVP(nn.Module):
    # 定义初始化方法
    def __init__(self, input_dim, condition_dim, num_couplings=8):
        super(ConditionalRealNVP, self).__init__()
        self.num_couplings = num_couplings

        # 创建耦合层的列表
        self.couplings = nn.ModuleList([AffineCouplingLayer(input_dim, condition_dim) for _ in range(num_couplings)])

    # 定义forward方法
    def forward(self, x, condition):
        for i, coupling in enumerate(self.couplings):
            x = coupling(x, condition)
            if i < self.num_couplings - 1:
                x = x.flip(dims=[1])  # 在每个耦合层之后除了最后一个外翻转维度
        return x

    # 定义inverse方法
    def inverse(self, y, condition):
        for i, coupling in reversed(list(enumerate(self.couplings))):
            y = coupling.inverse(y, condition)
            if i > 0:
                y = y.flip(dims=[1])  # 在每个耦合层之前除了第一个外翻转维度
        return y


# 定义AudioVAE类
class AudioVAE_RealNVP(nn.Module):
    # 定义初始化方法
    def __init__(self, input_shape=(128, 44), latent_dim=20, condition_dim=192):
        super(AudioVAE_RealNVP, self).__init__()
        # ConditionalRealNVP
        self.realnvp = ConditionalRealNVP(input_dim=20, condition_dim=192)  # RealNVP模型

        self.input_shape = input_shape  # 例如 (128, 44) 对应梅尔频谱的形状
        self.latent_dim = latent_dim

        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_shape[0] * input_shape[1], 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
        )
        self.mu_layer = nn.Linear(256, latent_dim)  # 均值向量
        self.logvar_layer = nn.Linear(256, latent_dim)  # 对数方差向量

        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_shape[0] * input_shape[1]),
            nn.Sigmoid(),  # 使用Sigmoid激活函数确保输出值在0到1之间
        )

    # 定义encode方法
    def encode(self, x):
        x = x.reshape(-1, self.input_shape[0] * self.input_shape[1])  # 重塑输入以匹配encoder的期望输入尺寸
        x = self.encoder(x)
        mu = self.mu_layer(x)
        logvar = self.logvar_layer(x)
        return mu, logvar

    # 定义reparameterize方法
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    # 定义decode方法
    def decode(self, z):
        return self.decoder(z).view(-1, *self.input_shape)

    # 定义forward方法
    def forward(self, x, source_condition, reference_condition, mode='train'):
        if mode == 'valid':
            min_val = x.min().item()  # 获取原始音频数据的最小值
            max_val = x.max().item()  # 获取原始音频数据的最大值

            x = normalize_tensor_0_1(x)
            source_condition = normalize_tensor_0_1(source_condition)
            reference_condition = normalize_tensor_0_1(reference_condition)

            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_transformed = self.realnvp(z, source_condition)  # 应用RealNVP转换

            z_inversed = self.realnvp.inverse(z_transformed, reference_condition)
            generated_x = self.decode(z_inversed[:, :self.latent_dim])  # 仅使用转换后的z的前半部分进行解码
            generated_x = denormalize_tensor_0_1(generated_x, min_val, max_val)
            return generated_x
        else:
            x = normalize_tensor_0_1(x).squeeze(1)
            source_condition = normalize_tensor_0_1(source_condition).squeeze(1)
            reference_condition = normalize_tensor_0_1(reference_condition).squeeze(1)

            mu, logvar = self.encode(x)
            z = self.reparameterize(mu, logvar)
            z_transformed = self.realnvp(z, source_condition)  # 应用RealNVP转换

            z_inversed = self.realnvp.inverse(z_transformed, reference_condition)
            generated_x = self.decode(z_inversed[:, :self.latent_dim])  # 仅使用转换后的z的前半部分进行解码
            return generated_x, x, mu, logvar, z, z_inversed


def loss_function(recon_x, x, mu, logvar, z, z_inversed, lambda_recon=0.1, lambda_kl=0.1, lambda_nvp=0.1):
    # print(recon_x.size(), x.size(), mu.size(), logvar.size(), z.size(), z_inversed.size())
    # 重建损失: 比较重建的音频和原始音频
    recon_loss = F.mse_loss(recon_x, x)

    # KL散度损失: 鼓励隐变量接近标准正态分布
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    # RealNVP特定的损失: 可以是z_transformed和z_inversed之间的差异，即循环一致性损失
    nvp_loss = F.mse_loss(z_inversed, z)

    # 总损失
    total_loss = lambda_recon * recon_loss + lambda_kl * kl_loss + lambda_nvp * nvp_loss
    # exit()
    return total_loss, recon_loss, kl_loss, nvp_loss
