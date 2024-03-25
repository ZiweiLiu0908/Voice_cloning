import torch
import torch.nn as nn


class WaveNetBlock(nn.Module):
    def __init__(self, in_channels=192, out_channels=192, num_blocks=8):
        super(WaveNetBlock, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=2 ** i, dilation=2 ** i),
                nn.Tanh()
            ) for i in range(num_blocks)
        ])

    def forward(self, x):
        for block in self.blocks:
            x = x + block(x)  # Residual connection
        return x


class FlowModel(nn.Module):
    def __init__(self):
        super(FlowModel, self).__init__()
        self.conv1d = nn.Conv1d(96, 192, kernel_size=1)  # Assuming kernel size of 1 for simplicity
        self.waveNet = WaveNetBlock()  # Placeholder for the WaveNet-like block
        self.conv1d_g = nn.Conv1d(96, 192, kernel_size=1)  # g convolution
        self.conv1d_m = nn.Conv1d(192, 96, kernel_size=1)  # m convolution, mean only

    def forward(self, x, mode='forward'):
        if mode == 'forward':
            x = x.permute(0, 2, 1)
            # Assuming x is [B, 1, 192]
            x0, x1 = x.split([96, 96], dim=1)  # Split into two parts

            # Process x0 through conv1d to get g
            g = self.conv1d(x0)
            g = self.waveNet(g)  # Pass through WaveNet-like block

            # Process x1 through another conv1d to get m
            m = self.conv1d_m(g)

            # Element-wise multiplication and exponentiation with log_std
            x1_exp = x1

            # Combine m and x1_exp
            z = m + x1_exp

            # Concatenate g and z to get z_p
            z_p = torch.cat((x0, z), dim=1)

            return z_p
        elif mode == 'inverse':
            x = x.permute(0, 2, 1)
            # Assuming x is [B, 1, 192]
            x0, x1 = x.split([96, 96], dim=1)  # Split into two parts

            # Process x0 through conv1d to get g
            g = self.conv1d(x0)
            g = self.waveNet(g)  # Pass through WaveNet-like block

            # Process x1 through another conv1d to get m
            m = self.conv1d_m(g)

            # Combine m and x1_exp
            z = x1 - m

            # Concatenate g and z to get z_p
            z_p = torch.cat((x0, z), dim=1)

            return z_p

