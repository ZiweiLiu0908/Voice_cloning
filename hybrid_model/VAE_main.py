import torch.nn
import torch.nn.functional as F

from main.Flow import FlowModel
from main.extractor.text_feat_model import TextFeat
from main.extractor.tone_feat_model import ToneFeat

import torch
import torch.nn as nn


class WaveNetBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation_rate, causal=True):
        super(WaveNetBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation_rate, padding=0)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, padding=0)
        self.causal = causal
        self.gate = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.filter = nn.Conv1d(out_channels, out_channels, kernel_size=1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        if self.causal:
            x = self.conv1(F.pad(x, (self.conv1.dilation[0] * (self.conv1.kernel_size[0] - 1), 0)))
        else:
            x = self.conv1(x)
        x = self.tanh(x)
        # x = x.permute(0, 2, 1)
        y = self.conv2(x)
        g = self.gate(x)
        f = self.filter(x)
        y = torch.tanh(f) * y + (1 - torch.tanh(g)) * x

        return y



class AudioVAE_Flow(nn.Module):

    def __init__(self, device='cuda:0'):
        super(AudioVAE_Flow, self).__init__()

        self.device = device


        self.tone_mu_layer = nn.Linear(512, 192).to(device)
        self.tone_logvar_layer = nn.Linear(512, 192).to(device)

        self.text_mu_layer = nn.Linear(512, 192).to(device)
        self.text_logvar_layer = nn.Linear(512, 192).to(device)


        self.tone_extractor = ToneFeat().to(device)
        checkpoint = torch.load('extractor/extractor_pth/toneFeat_model_checkpoint.pth.tar', map_location=device)
        self.tone_extractor.load_state_dict(checkpoint['state_dict'])

        self.text_extractor = TextFeat().to(device)
        checkpoint = torch.load('extractor/extractor_pth/textFeat_model_checkpoint.pth.tar', map_location=device)
        self.text_extractor.load_state_dict(checkpoint['state_dict'])

        self.flow = FlowModel().to(device)


        self.encode_conv1 = nn.Conv1d(4096, 2048, kernel_size=1).to(device)
        self.encode_wavenet = WaveNetBlock(in_channels=2048, out_channels=2048, kernel_size=3, dilation_rate=1).to(device)
        self.encode_conv2 = nn.Conv1d(2048, 1024, kernel_size=1).to(device)
        self.encode_conv3 = nn.Conv1d(1024, 192 * 2, kernel_size=1).to(device)

        self.encoder_mu_layer = nn.Linear(192, 192).to(device)
        self.encoder_logvar_layer = nn.Linear(192, 192).to(device)


        self.decode_scale = nn.Linear(192, 192).to(device)
        self.decode_bias = nn.Linear(192, 192).to(device)
        self.decode_pre_conv1 = nn.Conv1d(192, 320, kernel_size=1).to(device)
        self.decode_ConvTranspose1d_1 = nn.ConvTranspose1d(in_channels=320, out_channels=160, kernel_size=5, stride=5, padding=0).to(device)
        self.decode_ConvTranspose1d_2 = nn.ConvTranspose1d(in_channels=512, out_channels=1024, kernel_size=1, stride=1, padding=0).to(device)
        self.decode_wavenet1 = WaveNetBlock(in_channels=160, out_channels=512, kernel_size=3, dilation_rate=1).to(device)
        self.decode_wavenet2 = WaveNetBlock(in_channels=1024, out_channels=4096, kernel_size=3, dilation_rate=1).to(device)
        self.decode_conv = nn.Conv1d(4096, 4096, kernel_size=1).to(device)




        self.low_level_fusion = nn.Linear(192 + 192, 512)

        self.high_level_fusion = nn.Linear(
            192*4, 192)

        self.high_fusion_mu = nn.Linear(192*2,192)
        self.high_fusion_std = nn.Linear(192 * 2, 192)

    def extractor(self, source_text_feature, source_tone_feature, source_vec_feature,
                  reference_text_feature, reference_tone_feature, reference_vec_feature):
        tone_feat = self.tone_extractor(reference_text_feature, reference_tone_feature, reference_vec_feature)
        text_feat = self.text_extractor(source_text_feature, source_tone_feature, source_vec_feature)
        low_level_fused = self.low_level_fusion(torch.cat([tone_feat, text_feat], dim=-1))

        tone_mu = self.tone_mu_layer(low_level_fused)
        tone_logvar = self.tone_logvar_layer(low_level_fused)
        text_mu = self.text_mu_layer(low_level_fused)
        text_logvar = self.text_logvar_layer(low_level_fused)

        z_p = self.high_level_fusion(torch.cat([tone_mu, tone_logvar, text_mu, text_logvar], dim=-1))
        z_p_mu = self.high_fusion_mu(torch.cat([tone_mu, text_mu], dim=-1))
        z_p_log_var = self.high_fusion_std(torch.cat([tone_logvar, text_logvar], dim=-1))




        return z_p, z_p_mu, z_p_log_var

    def encode(self, x):
        x = x.permute(0, 2, 1)

        x = self.encode_conv1(x)
        x = self.encode_wavenet(x)
        x = self.encode_conv2(x)
        x = self.encode_conv3(x)
        x = x.squeeze(2)
        x = torch.chunk(x, chunks=2, dim=1)
        z_q_mu = self.encoder_mu_layer(x[0])
        z_q_log_var = self.encoder_logvar_layer(x[1])
        z_q = self.reparameterize(z_q_mu, z_q_log_var)
        return z_q.unsqueeze(0), z_q_mu.unsqueeze(0), z_q_log_var.unsqueeze(0)


    def reparameterize(self, mu, logvar):
        logvar = logvar.clamp(-10, 10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std


    def decode(self, z, spk):
        z = (z - z.mean()) / z.std()
        spk = spk.unsqueeze(2)
        z = z.permute(0, 2, 1)
        spk = spk.permute(0, 2, 1)
        z = z * self.decode_scale(spk) + self.decode_bias(spk)
        z = z.permute(0, 2, 1)
        z = self.decode_pre_conv1(z)
        t = torch.nn.Softplus()(z)
        z = F.tanh(t)

        z = self.decode_ConvTranspose1d_1(z)
        z = self.decode_wavenet1(z)

        z = self.decode_ConvTranspose1d_2(z)
        z = self.decode_wavenet2(z)

        z = F.tanh(self.decode_conv(z))

        return z.mean(dim=2, keepdim=True)


    def forward(self, target_audio, source_text_feature, source_tone_feature, source_vec_feature,
                reference_text_feature, reference_tone_feature, reference_vec_feature, mode='train'):
        if mode == 'train':

            z_q, z_q_mu, z_q_log_var = self.encode(target_audio)


            z_p, z_p_mu, z_p_log_var = self.extractor(source_text_feature, source_tone_feature,
                                                      source_vec_feature,
                                                      reference_text_feature, reference_tone_feature,
                                                      reference_vec_feature)

            z_q = (z_q - z_q.mean()) / z_q.std()
            z_p = (z_p - z_p.mean()) / z_p.std()

            # flow forward to get z_p_head
            z_p_head = self.flow(z_q, mode='forward')
            # flow inverse to get z_q_head
            z_q_head = self.flow(z_p, mode='inverse')

            # regenerate audio
            generated_audio = self.decode(z_q_head, reference_tone_feature)
            return z_q, z_p, z_q_head.permute(0, 2, 1), z_p_head.permute(0, 2, 1), z_q_mu, z_q_log_var, z_p_mu, z_p_log_var, generated_audio.permute(0, 2, 1)
        else:
            z_p, z_p_mu, z_p_log_var = self.extractor(source_text_feature, source_tone_feature,
                                                      source_vec_feature,
                                                      reference_text_feature, reference_tone_feature,
                                                      reference_vec_feature)
            z_p = (z_p - z_p.mean()) / z_p.std()
            # flow inverse to get z_q_head
            z_q_head = self.flow(z_p, mode='inverse')
            # regenerate audio
            generated_audio = self.decode(z_q_head, reference_tone_feature)
            return generated_audio