import torch.nn as nn


class HeadBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(HeadBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, stride=2, padding=1)
        self.prelu = nn.PReLU()

    def forward(self, x):
        # print("Head Part")
        # print(x.size)
        out = self.prelu(self.conv(x))
        # print(out.size())
        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, use_residual=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channel, in_channel, kernel_size=kernel_size, padding=1, padding_mode='reflect')
        self.bn1 = nn.BatchNorm2d(in_channel)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1, padding_mode='reflect')
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.prelu2 = nn.PReLU()
        self.use_residual = use_residual

    def forward(self, x):
        # print("Res Part")
        # print(x.size())
        out = self.prelu1(self.bn1(self.conv1(x)))
        # print(out.size())
        out = self.prelu2(self.bn2(self.conv2(out)))
        # print(out.size())
        if self.use_residual:
            out = x + out
        return out


class TailBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(TailBlock, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size, padding=1, padding_mode='reflect')
        self.bn = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        # print("Tail Part")
        # print(x.size())
        out = self.bn(self.conv(x))
        # print(out.size())
        return out


class UpSample(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size):
        super(UpSample, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size, padding=1, padding_mode="reflect"),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.PReLU()
        )

    def forward(self, x):
        # print("UpSample Part")
        # print(x.size())
        out = self.upsample(x)
        # print(out.size())
        return out
