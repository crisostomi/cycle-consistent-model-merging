import torch.nn as nn
import torch.nn.functional as F

from ccmm.models.utils import LayerNorm2d

cfg = {
    "VGG11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "VGG16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "VGG19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, num_classes, classifier_width=512, in_channels=3):
        super(VGG, self).__init__()
        self.in_channels = in_channels
        self.embedder = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Sequential(
            *[
                nn.Linear(classifier_width, classifier_width),
                nn.ReLU(),
                nn.Linear(classifier_width, classifier_width),
                nn.ReLU(),
                nn.Linear(classifier_width, num_classes),
            ]
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)

        out = self.embedder(x)

        out = out.view(out.size(0), -1)

        out = self.classifier(out)

        return F.log_softmax(out, dim=1)

    def _make_layers(self, cfg):
        layers = []
        in_channels = self.in_channels
        for x in cfg:
            if x == "M":
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif isinstance(x, int):
                layers += [
                    nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                    LayerNorm2d(x),
                    nn.ReLU(),
                ]
                in_channels = x
            else:
                raise ValueError("Unknown layer type: {}".format(x))

        return nn.Sequential(*layers)
