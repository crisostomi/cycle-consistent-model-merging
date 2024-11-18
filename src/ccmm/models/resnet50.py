import timm
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
from transformers import AutoImageProcessor, ResNetForImageClassification


class ResNet50(nn.Module):
    def __init__(self, num_classes):
        super(ResNet50, self).__init__()

        self.num_classes = num_classes

        self.model = resnet50(num_classes=num_classes)

    def forward(self, x):
        return self.model(x)


class PretrainedResNet50(nn.Module):
    def __init__(self, num_classes, weights: str):
        super(PretrainedResNet50, self).__init__()

        self.num_classes = num_classes

        # if weights == "v1":
        #     weights = ResNet50_Weights.IMAGENET1K_V1
        #     model = resnet50(num_classes=num_classes, weights=weights)
        # elif weights == "v2":
        #     weights = ResNet50_Weights.IMAGENET1K_V2
        #     model = resnet50(num_classes=num_classes, weights=weights)
        # if weights == "v1":
        #     model = timm.create_model("resnet50.gluon_in1k", pretrained=True)
        # elif weights == "v2":
        #     model = timm.create_model("resnet50.ram_in1k", pretrained=True)
        # elif weights == "v3":
        #     model = timm.create_model("resnet50.a1_in1k", pretrained=True)
        model = timm.create_model(f"resnet50.{weights}_in1k", pretrained=True)
        # elif weights == "v3":
        # model = timm.create_model("resnet50.a1_in1k", pretrained=True)
        # elif weights == 'v3': # a1_in1k, a2_in1k, a1h_in1k, ram_in1k, ra_in1k, c1_in1k, gluon_in1k
        # model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

        self.model = model

    def forward(self, x):
        return self.model(x)
