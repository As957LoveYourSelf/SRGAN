from blocks import ResidualBlock, HeadBlock, TailBlock, UpSample
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, residual_num: int, upsample_x: int):
        assert (upsample_x >= 1)
        assert (residual_num >= 1)
        super(Generator, self).__init__()
        # define down_sample layer
        self.hb = HeadBlock(in_channel=3, out_channel=64, kernel_size=9)
        # define residual blocks
        self.residual_blocks = []
        for i in range(residual_num):
            self.residual_blocks.append(ResidualBlock(64, 64, kernel_size=3))
        self.residuals = nn.Sequential(*self.residual_blocks)
        self.tailblock = TailBlock(in_channel=64, out_channel=64, kernel_size=3)
        # define residual blocks
        self.upsample_blocks = []
        in_channel = 64
        out_channel = 256
        for i in range(upsample_x - 1):
            self.upsample_blocks.append(UpSample(in_channel=in_channel, out_channel=out_channel, kernel_size=3))
            in_channel = out_channel
        self.upsample_blocks.append(UpSample(in_channel=out_channel, out_channel=out_channel // 2, kernel_size=3))
        self.upsamples = nn.Sequential(*self.upsample_blocks)
        # the last convolution layer to rebuilt image
        self.conv = nn.Conv2d(in_channels=out_channel // 2, out_channels=3, kernel_size=9)

    def forward(self, x):
        hb_out = self.hb(x)
        out = self.residuals(hb_out)
        out = self.tailblock(out)
        out = hb_out + out
        out = self.upsamples(out)
        out = self.conv(out)
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # input shape (3, 96, 96)
        self.conv_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(64, 96, 96)
            nn.Conv2d(64, 64, kernel_size=4, stride=2),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(64, 48, 48)
            nn.Conv2d(64, 128, kernel_size=3, stride=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(128, 48, 48)
            nn.Conv2d(128, 128, kernel_size=4, stride=2),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(128, 24, 24)
            nn.Conv2d(128, 256, kernel_size=3, stride=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(256, 24, 24)
            nn.Conv2d(256, 256, kernel_size=4, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(256, 12, 12)
            nn.Conv2d(256, 512, kernel_size=3, stride=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(512, 12, 12)
            nn.Conv2d(512, 512, kernel_size=4, stride=2),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # output shape (512, 6, 6)
            nn.Flatten()
        )
        self.liner1 = nn.Linear(512 * 6 * 6, 1024)
        self.liner2 = nn.Linear(1024, 1)
        self.activation = nn.Sigmoid()

    def forward(self, x):
        out = self.conv_model(x)
        out = self.liner1(out)
        out = self.liner2(out)
        out = self.activation(out)
        return out
