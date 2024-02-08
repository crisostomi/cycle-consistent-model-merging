import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


class ResNet18PreTrained(nn.Module):
    def __init__(self, num_classes):
        super(ResNet18PreTrained, self).__init__()

        self.num_classes = num_classes

        self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

        self.model.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.model(x)
