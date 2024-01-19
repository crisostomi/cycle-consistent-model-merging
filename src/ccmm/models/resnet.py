import logging

import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

pylogger = logging.getLogger(__name__)


def apply_modules(x, fs):
    for f in fs:
        x = f(x)
    return x


class LayerNorm2d(nn.Module):
    def __init__(self, num_features):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.GroupNorm(num_groups=1, num_channels=num_features)

    def forward(self, x):

        return self.layer_norm(x)


class ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 32
        # standard (R, G, B)
        input_channels = 3

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        num_blocks_per_layer = (depth - 4) // 6

        out_channels = [16 * widen_factor, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # (16 * wm, input_channels, 3, 3)
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=out_channels[0], kernel_size=3, padding=1, bias=False
        )
        # (16 * wm)
        self.bn1 = LayerNorm2d(out_channels[0])

        strides = [1, 2, 2]

        for i in range(3):
            self.add_module(
                "blockgroup{}".format(i + 1),
                BlockGroup(
                    in_features=out_channels[i],
                    num_channels=out_channels[i + 1],
                    num_blocks=num_blocks_per_layer,
                    stride=strides[i],
                ),
            )

        self.linear = nn.Linear(out_channels[3], num_classes)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        # (B, 32, 32, 32)
        out = self.blockgroup1(out)
        # (B, 64, 32, 32)
        out = self.blockgroup2(out)
        # (B, 64, 16, 16)
        out = self.blockgroup3(out)

        out = reduce(out, "n c h w -> n c", "mean")

        out = self.linear(out)

        out = F.log_softmax(out, dim=1)

        return out


class BlockGroup(nn.Module):
    num_channels: int = None
    num_blocks: int = None
    stride: int = None
    in_features: int = None

    def __init__(self, in_features, num_channels, num_blocks, stride):
        super(BlockGroup, self).__init__()
        self.num_channels = num_channels
        self.num_blocks = num_blocks
        self.stride = stride
        self.in_features = in_features

        assert self.num_blocks > 0

        # this is how it's done in git-rebasin
        strides = [self.stride, 1, 1]

        for i in range(self.num_blocks):
            self.add_module(
                "block{}".format(i + 1),
                Block(self.in_features, self.num_channels, strides[i]),
            )
            self.in_features = self.num_channels

    def forward(self, x):
        return apply_modules(x, self.children())


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(Block, self).__init__()
        # input_dim = [batch_size, in_channels, dim, dim]
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = LayerNorm2d(out_channels)

        # input_dim = [planes, dim, dim]
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = LayerNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1:
            assert stride == 2

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
                LayerNorm2d(out_channels),
            )

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)

        out = F.relu(h + self.shortcut(x))

        return out
