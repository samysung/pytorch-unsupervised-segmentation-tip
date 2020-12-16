# CNN model
import torch.nn as nn
import torch.nn.functional as F


class MyNet(nn.Module):

    def __init__(self, input_dim, n_channel, n_conv):

        super(MyNet, self).__init__()
        self.n_conv = n_conv
        self.n_channel = n_channel
        self.conv1 = nn.Conv2d(input_dim, self.n_channel, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(self.n_channel)
        self.conv2 = nn.ModuleList()
        self.bn2 = nn.ModuleList()

        for i in range(self.n_conv-1):

            self.conv2.append(nn.Conv2d(n_channel, self.n_channel, kernel_size=3, stride=1, padding=1))
            self.bn2.append(nn.BatchNorm2d(self.n_channel))

        self.conv3 = nn.Conv2d(self.n_channel, self.n_channel, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(self.n_channel)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.bn1(x)

        for i in range(self.n_conv-1):

            x = self.conv2[i](x)
            x = F.relu(x)
            x = self.bn2[i](x)

        x = self.conv3(x)
        x = self.bn3(x)

        return x
