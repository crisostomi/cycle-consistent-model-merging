import logging

import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

pylogger = logging.getLogger(__name__)


class ResNet(nn.Module):
    def __init__(self, depth, widen_factor, num_classes):
        super(ResNet, self).__init__()
        self.in_planes = 16

        assert (depth - 4) % 6 == 0, "Wide-resnet depth should be 6n+4"
        num_blocks_per_layer = (depth - 4) / 6

        planes = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        spatial_dimensions = [32, 32, 32, 16]

        self.conv1 = conv3x3(3, planes[0])
        self.bn1 = nn.LayerNorm([planes[0], 32, 32])

        self.layer1 = self._wide_layer(
            wide_basic, planes=planes[1], num_blocks=num_blocks_per_layer, stride=1, spatial_dim=spatial_dimensions[1]
        )
        self.layer2 = self._wide_layer(
            wide_basic, planes=planes[2], num_blocks=num_blocks_per_layer, stride=2, spatial_dim=spatial_dimensions[2]
        )
        self.layer3 = self._wide_layer(
            wide_basic, planes=planes[3], num_blocks=num_blocks_per_layer, stride=2, spatial_dim=spatial_dimensions[3]
        )

        self.out_bn = nn.LayerNorm([planes[3], 8, 8])
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
        out = self.bn1(self.conv1(x))

        out = self.layer1(out)
        # (B, 32, 32, 32)
        out = self.layer2(out)
        # (B, 64, 16, 16)
        out = self.layer3(out)

        out = F.relu(self.out_bn(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        return out


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        init.xavier_uniform_(m.weight, gain=np.sqrt(2))
        init.constant_(m.bias, 0)
    elif classname.find("LayerNorm") != -1:
        init.constant_(m.weight, 1)
        init.constant_(m.bias, 0)


class wide_basic(nn.Module):
    def __init__(self, in_planes, planes, stride, spatial_dim):
        super(wide_basic, self).__init__()
        # input_dim = [in_planes, dim, dim]
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.LayerNorm([planes, spatial_dim, spatial_dim])

        # input_dim = [planes, dim, dim]
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.LayerNorm([planes, spatial_dim // stride, spatial_dim // stride])

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.LayerNorm([planes, spatial_dim // stride, spatial_dim // stride]),  # nn.GroupNorm(1, planes)
            )

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = F.relu(h)

        h = self.conv2(h)
        h = self.bn2(h)
        h = F.relu(h)

        out = h + self.shortcut(x)

        return out
