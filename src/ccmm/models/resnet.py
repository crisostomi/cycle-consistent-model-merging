import logging

import torch.nn as nn
import torch.nn.functional as F
from einops import reduce

pylogger = logging.getLogger(__name__)


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

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        num_blocks_per_layer = (depth - 4) / 6

        planes = [16 * widen_factor, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        spatial_dimensions = [32, 32, 32, 16]

        self.conv1 = nn.Conv2d(3, planes[0], kernel_size=3, padding=1, bias=False)
        self.bn1 = LayerNorm2d(planes[0])

        self.layer1 = self._wide_layer(
            wide_basic, planes=planes[1], num_blocks=num_blocks_per_layer, stride=1, spatial_dim=spatial_dimensions[1]
        )
        self.layer2 = self._wide_layer(
            wide_basic, planes=planes[2], num_blocks=num_blocks_per_layer, stride=2, spatial_dim=spatial_dimensions[2]
        )
        self.layer3 = self._wide_layer(
            wide_basic, planes=planes[3], num_blocks=num_blocks_per_layer, stride=2, spatial_dim=spatial_dimensions[3]
        )

        self.linear = nn.Linear(planes[3], num_classes)

    def _wide_layer(self, block, planes, num_blocks, stride, spatial_dim):
        strides = [stride] + [1] * (int(num_blocks) - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride, spatial_dim))
            spatial_dim = spatial_dim // stride
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))

        out = self.layer1(out)
        # (B, 32, 32, 32)
        out = self.layer2(out)
        # (B, 64, 16, 16)
        out = self.layer3(out)

        out = reduce(out, "n c h w -> n c", "mean")

        out = self.linear(out)

        out = F.log_softmax(out, dim=1)

        return out


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride, spatial_dim):
        super(wide_basic, self).__init__()
        # input_dim = [in_planes, dim, dim]
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = LayerNorm2d(planes)

        # input_dim = [planes, dim, dim]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = LayerNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:

            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False),
                LayerNorm2d(planes),
            )

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)

        out = F.relu(h + self.shortcut(x))

        return out
