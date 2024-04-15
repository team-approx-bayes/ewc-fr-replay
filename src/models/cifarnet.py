""" Adapted from https://github.com/team-approx-bayes/fromp/blob/main/models.py """

import torch
from torch import nn

class CifarNet(nn.Module):
    def __init__(self, d_in, d_out):
        """
            d_in: number of input channels
            d_out: number of output channels
        """

        super(CifarNet, self).__init__()

        self.conv_block = nn.Sequential(
            nn.Conv2d(d_in, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Dropout(p=0.25)
        )

        self.linear_block = nn.Sequential(
            nn.Linear(64*6*6, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.upper = nn.Linear(512, d_out)

    def forward(self, x):
        o = self.conv_block(x)
        o = torch.flatten(o, 1)
        o = self.linear_block(o)
        o = self.upper(o)
        return o
