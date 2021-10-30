from blocks import ResidualBlock, HeadBlock, TailBlock, UpSample
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, residual_num: int, upsample_x: int):
        assert (upsample_x >= 1)
        assert (residual_num >= 1)
        super(Generator, self).__init__()
        # define down_sample layer
        self.hb = HeadBlock(in_channel=3, out_channel=64, kernel_size=4)
        # define residual blocks
        self.residual_blocks = []
        for i in range(residual_num):
            self.residual_blocks.append(ResidualBlock(64, 64, kernel_size=3))
        self.residuals = nn.Sequential(*self.residual_blocks)
        self.tailblock = TailBlock(in_channel=64, out_channel=64, kernel_size=3)
        # define UpSample blocks
        self.upsamples = nn.Sequential(
            UpSample(in_channel=64, out_channel=128, kernel_size=3),
            UpSample(in_channel=128, out_channel=256, kernel_size=3),
            UpSample(in_channel=256, out_channel=256, kernel_size=3)
        )

        self.features = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=256, out_channels=128, kernel_size=3),
            # the last convolution layer to rebuilt image
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=128, out_channels=64, kernel_size=3),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=64, out_channels=3, kernel_size=3)
        )

    def forward(self, x):
        hb_out = self.hb(x)
        out = self.residuals(hb_out)
        out = self.tailblock(out)
        out = hb_out + out
        out = self.upsamples(out)
        out = self.features(out)
        # print(out.size())
        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        # input shape (3, 96, 96)
        self.conv_model = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(64, 96, 96)
            nn.Conv2d(64, 64, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(64, 48, 48)
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(128, 48, 48)
            nn.Conv2d(128, 128, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(128, 24, 24)
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(256, 24, 24)
            nn.Conv2d(256, 256, kernel_size=4, stride=2, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # feature shape(256, 12, 12)
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 128, kernel_size=3, stride=1, padding=1, padding_mode="reflect"),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )

        self.predict = nn.Sequential(
            nn.Conv2d(128, 1, kernel_size=3, padding=1, padding_mode="reflect")
        )

        self.activation = nn.Sigmoid()

    def forward(self, x):
        # print(x.size())
        out = self.conv_model(x)
        # print(out.size())
        self.predict(out)
        # print(out.size())
        out = self.activation(out)
        # print(out.size())
        return out
