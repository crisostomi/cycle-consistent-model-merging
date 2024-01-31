import copy

import torch
import torch.nn as nn

from ccmm.models.repaired_resnet import RepairedResNet
from ccmm.models.resnet import ResNet
from ccmm.utils.utils import fuse_batch_norm_into_conv


def replace_conv_layers(module):
    for name, child in module.named_children():
        if isinstance(child, nn.Conv2d):
            setattr(module, name, ResetConv(child))
        else:
            replace_conv_layers(child)


def make_tracked_net(model):
    """
    Wraps each convolutional layer in a ResetConv module.
    """
    tracked_model = copy.deepcopy(model)
    if isinstance(tracked_model.model, ResNet):
        tracked_model = add_resnet_junctures(tracked_model)
    replace_conv_layers(tracked_model)

    return tracked_model.eval()


def reset_bn_stats(model, epochs, loader):
    """
    Reset batchnorm stats. We use the train loader with data augmentation as this gives better results.
    """
    # resetting stats to baseline first as below is necessary for stability
    for m in model.modules():
        if type(m) == nn.BatchNorm2d:
            m.momentum = None  # use simple average
            m.reset_running_stats()

    # run a single train epoch with augmentations to recalc stats
    model.train()
    for _ in range(epochs):
        with torch.no_grad():
            for images, _ in loader:
                _ = model(images.cuda())


def compute_goal_statistics_two_models(model_a, model_to_repair, model_b):
    """
    Set the goal mean/std in added bns of interpolated network, and turn batch renormalization on
    """
    for m_a, m_interp, m_b in zip(model_a.modules(), model_to_repair.modules(), model_b.modules()):

        if not isinstance(m_a, ResetConv):
            continue

        # get goal statistics -- interpolate the mean and std of parent networks
        mu_a = m_a.bn.running_mean
        mu_b = m_b.bn.running_mean
        goal_mean = (mu_a + mu_b) / 2

        var_a = m_a.bn.running_var
        var_b = m_b.bn.running_var
        goal_var = ((var_a.sqrt() + var_b.sqrt()) / 2).square()

        # set these in the interpolated bn controller
        m_interp.set_stats(goal_mean, goal_var)

        # turn rescaling on
        m_interp.rescale = True


def compute_goal_statistics(model_to_repair, endpoint_models):
    """
    Set the goal mean/std in added bns of interpolated network, and turn batch renormalization on
    """

    for m_interp, *endpoint_modules in zip(model_to_repair.modules(), *[model.modules() for model in endpoint_models]):

        if not isinstance(m_interp, ResetConv):
            continue

        mu_endpoints = torch.stack([m.bn.running_mean for m in endpoint_modules])

        goal_mean = mu_endpoints.mean(dim=0)

        var_endpoints = torch.stack([m.bn.running_var for m in endpoint_modules])

        goal_var = var_endpoints.mean(dim=0)

        # set these in the interpolated bn controller
        m_interp.set_stats(goal_mean, goal_var)

        # turn rescaling on
        m_interp.rescale = True


def repair_model(model_to_repair, models, train_loader):

    model_to_repair = copy.deepcopy(model_to_repair)

    repaired_model = make_tracked_net(model_to_repair).cuda()

    wrapped_models = [make_tracked_net(model).cuda() for model in models.values()]

    for model in wrapped_models:
        reset_bn_stats(model.cuda(), loader=train_loader, epochs=1)

    compute_goal_statistics(repaired_model, wrapped_models)

    reset_bn_stats(repaired_model.cuda(), loader=train_loader, epochs=1)

    repaired_model = fuse_tracked_net(repaired_model)

    if isinstance(model_to_repair.model, ResNet):
        kwargs = {
            "depth": repaired_model.model.depth,
            "widen_factor": repaired_model.model.widen_factor,
            "num_classes": repaired_model.model.num_classes,
        }
        repaired_model_correct_class = RepairedResNet(**kwargs).cuda().eval()
        repaired_model_correct_class.load_state_dict(repaired_model.model.state_dict())

        model_to_repair.model = repaired_model_correct_class
        model_to_repair.hparams["model"]["_target_"] = "ccmm.models.repaired_resnet.RepairedResNet"

        return model_to_repair

    return repaired_model


def fuse_batch_norm_into_conv_recursive(module):

    for name, child in module.named_children():
        if isinstance(child, ResetConv):
            conv = fuse_batch_norm_into_conv(child.conv, child.bn)

            setattr(module, name, conv)
        else:
            fuse_batch_norm_into_conv_recursive(child)


def fuse_tracked_net(tracked_model):

    fuse_batch_norm_into_conv_recursive(tracked_model.model)

    return tracked_model


def add_resnet_junctures(model):
    tracked_model = copy.deepcopy(model)

    blocks = [
        *tracked_model.model.blockgroup1.children(),
        *tracked_model.model.blockgroup2.children(),
        *tracked_model.model.blockgroup3.children(),
    ]

    for block in blocks:

        if len(block.shortcut) > 0:
            continue

        planes = len(block.bn2.layer_norm.weight)

        shortcut = nn.Conv2d(planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        shortcut.weight.data[:, :, 0, 0] = torch.eye(planes)

        block.shortcut = shortcut

    return tracked_model.cuda().eval()


class TrackLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)

    def get_stats(self):
        return (self.bn.running_mean, self.bn.running_var.sqrt())

    def forward(self, x):
        x1 = self.layer(x)
        self.bn(x1)
        return x1


class ResetLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.layer = layer
        self.bn = nn.BatchNorm2d(layer.out_channels)

    def set_stats(self, goal_mean, goal_std):
        self.bn.bias.data = goal_mean
        self.bn.weight.data = goal_std

    def forward(self, x):
        x1 = self.layer(x)
        return self.bn(x1)


class ResetConv(nn.Module):
    def __init__(self, conv):
        super().__init__()
        self.out_channels = conv.out_channels
        self.conv = conv
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.rescale = False

    def set_stats(self, goal_mean, goal_var, eps=1e-5):
        self.bn.bias.data = goal_mean
        goal_std = (goal_var + eps).sqrt()
        self.bn.weight.data = goal_std

    def forward(self, x):
        x = self.conv(x)
        if self.rescale:
            x = self.bn(x)
        else:
            self.bn(x)
        return x
