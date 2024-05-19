import torch
import torch.nn as nn


class MLP(nn.Module):
    def __init__(self, input_shape=28 * 28, num_classes=10, hidden_dim=512):
        super().__init__()
        self.input_shape = torch.prod(torch.tensor(list(input_shape)))
        self.layer0 = nn.Linear(self.input_shape, hidden_dim)
        self.layer1 = nn.Linear(hidden_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.layer4 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        x = x.view(-1, self.input_shape)
        x = nn.functional.relu(self.layer0(x))
        x = nn.functional.relu(self.layer1(x))
        x = nn.functional.relu(self.layer2(x))
        x = nn.functional.relu(self.layer3(x))
        x = self.layer4(x)

        return nn.functional.log_softmax(x, dim=-1)
