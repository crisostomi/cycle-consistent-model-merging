import torch.nn as nn


class LayerNorm2d(nn.Module):
    """
    Mimics JAX's LayerNorm. This is used in place of BatchNorm to mimic Git Re-basin's setting as close as possible.
    """

    def __init__(self, num_features):
        super(LayerNorm2d, self).__init__()
        self.layer_norm = nn.LayerNorm((num_features,))

    def forward(self, x):
        x = x.permute(0, 2, 3, 1)

        x = self.layer_norm(x)

        x = x.permute(0, 3, 1, 2)
        return x
