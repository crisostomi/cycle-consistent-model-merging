import torch.nn as nn
from torch.nn.functional import relu


class MLP(nn.Module):
    def __init__(self, input=28 * 28, num_classes=10):
        super().__init__()
        self.input = input
        self.layer0 = nn.Linear(input, 512)
        self.layer1 = nn.Linear(512, 512)
        self.layer2 = nn.Linear(512, 512)
        self.layer3 = nn.Linear(512, 256)
        self.layer4 = nn.Linear(256, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input)
        x = relu(self.layer0(x))
        x = relu(self.layer1(x))
        x = relu(self.layer2(x))
        x = relu(self.layer3(x))
        x = self.layer4(x)

        return nn.functional.log_softmax(x)
