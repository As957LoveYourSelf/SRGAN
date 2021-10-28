import torch.nn as nn


class HeadBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(HeadBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.prelu(self.conv(x))
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, use_residual=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size)
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.prelu2 = nn.PReLU()
        self.use_residual = use_residual

    def forward(self, x):
        out = self.prelu1(self.bn1(self.conv1(x)))
        out = self.prelu2(self.bn2(self.conv2(out)))
        if self.use_residual:
            out = x + out
        return out


class TailBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(TailBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size)
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.bn(self.conv(x))
        return out


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(UpSample, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size)
        self.pixeshuffle = nn.PixelShuffle(2)
        self.prelu = nn.PReLU()

    def forward(self, x):
        out = self.prelu(self.pixeshuffle(self.conv(x)))
        return out
