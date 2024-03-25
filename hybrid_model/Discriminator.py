import torch
import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=1, padding=7),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=41, stride=4, padding=20, groups=4),
            nn.LeakyReLU(0.2),
            nn.Conv1d(64, 128, kernel_size=41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 256, kernel_size=41, stride=4, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(256, 512, kernel_size=41, stride=4, padding=20),
            nn.LeakyReLU(0.2),
            nn.Conv1d(512, 1, kernel_size=9, stride=1, padding=4),
        )

    def forward(self, x):
        x = self.disc(x)
        x = torch.flatten(x, 1)
        return x
