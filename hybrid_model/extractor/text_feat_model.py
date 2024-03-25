import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x):

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):

        return -grad_output



def grad_reverse(x):
    return GradientReversalFn.apply(x)


class WeightNorm(nn.Module):
    def __init__(self, module, dim=None):
        super(WeightNorm, self).__init__()
        self.module = module
        if dim is None:
            dim = 0
        self.dim = dim
        self.weight_g = nn.Parameter(torch.ones(module.weight.size(dim), 1, 1))
        self.bias_g = nn.Parameter(torch.zeros(module.bias.size()))

        self.weight_v = module.weight.detach() / torch.norm(module.weight.detach(), dim=self.dim, keepdim=True)

    def forward(self, x):
        self.weight_v = self.weight_v
        weight = self.weight_g * self.weight_v
        bias = self.bias_g + self.module.bias
        return F.conv1d(x, weight, bias, self.module.stride, self.module.padding, self.module.dilation)


class TextFeat(nn.Module):
    def __init__(self, num_heads=8, dim_model=192, device='cpu'):
        super(TextFeat, self).__init__()
        self.conv1 = nn.Conv1d(1280, 512, kernel_size=3, padding=1)
        self.conv1_1 = nn.Conv1d(512, 192, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(192, 192, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(512, 256, kernel_size=3, padding=1)
        self.conv3_1 = nn.Conv1d(256, 192, kernel_size=3, padding=1)

        self.mha = nn.MultiheadAttention(dim_model, num_heads, batch_first=True)


        self.text_conv_layers = nn.Sequential(
            WeightNorm(nn.Conv1d(192, 256, 3, padding=1)),
            nn.ReLU(),
            WeightNorm(nn.Conv1d(256, 512, 3, padding=1)),
            nn.ReLU(),
            WeightNorm(nn.Conv1d(512, 1280, 3, padding=1)),
            nn.ReLU()
        ).to(device=device)

        self.tone_conv_layers = nn.Sequential(
            WeightNorm(nn.Conv1d(dim_model, dim_model, 3, padding=1)),
            nn.ReLU(),
            WeightNorm(nn.Conv1d(dim_model, dim_model, 3, padding=1)),
            nn.ReLU(),
            WeightNorm(nn.Conv1d(dim_model, dim_model, 3, padding=1)),
            nn.ReLU()
        ).to(device)

    def forward(self, text_feature, tone_feature, vec_feature, mode='valid'):

        text_feature = text_feature.unsqueeze(0).transpose(1, 2)
        text_feature = self.conv1(text_feature)
        text_feature = F.relu(text_feature)
        text_feature = self.conv1_1(text_feature)
        text_feature = F.relu(text_feature)


        tone_feature = tone_feature.unsqueeze(0).transpose(1, 2)
        tone_feature = self.conv2(tone_feature)
        tone_feature = F.relu(tone_feature)


        vec_feature = vec_feature.unsqueeze(0).transpose(1, 2)
        vec_feature = self.conv3(vec_feature)
        vec_feature = F.relu(vec_feature)
        vec_feature = self.conv3_1(vec_feature)
        vec_feature = F.relu(vec_feature)


        x, _ = self.mha(text_feature.permute(0, 2, 1), tone_feature.permute(0, 2, 1), vec_feature.permute(0, 2, 1))
        if mode == 'valid':
            return x
        #######################################################################################################################################
        text = x.permute(0, 2, 1)
        text = self.text_conv_layers(text)
        text = text.squeeze(2)

        tone = x.permute(0, 2, 1)
        tone = grad_reverse(tone)
        tone = self.tone_conv_layers(tone)
        tone = tone.squeeze(2)
        return text, tone
