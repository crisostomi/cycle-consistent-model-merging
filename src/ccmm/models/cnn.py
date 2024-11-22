import torch
import torch.nn as nn
import torch.nn.functional as F


class Shortcut(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.identity = nn.Parameter(torch.eye(dim), requires_grad=False)

    def forward(self, x):
        return x @ self.identity.T


class CNN(nn.Module):
    def __init__(self, num_classes, hidden_dim, input):
        super(CNN, self).__init__()
        _, c, h, w = input
        self.conv1 = nn.Conv2d(c, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 128, 3, 1)

        self.shortcut = Shortcut(128)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.avg = nn.AvgPool2d(kernel_size=1, stride=1)

        conv_output_dim = 12 * 12 * 128
        self.fc1 = nn.Linear(conv_output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # b, c, h, w = x.size()

        x = self.conv1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = self.avg(x)

        x = self.shortcut(x.permute(0, 2, 3, 1))

        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.fc2(x)

        output = F.log_softmax(x, dim=1)
        return output
