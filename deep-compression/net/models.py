import torch.nn as nn
import torch.nn.functional as F

from .prune import PruningModule, MaskedLinear

class LeNet(PruningModule):
    def __init__(self, mask=False):
        super(LeNet, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.fc1 = linear(784, 300)
        self.fc2 = linear(300, 100)
        self.fc3 = linear(100, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


class LeNet_5(PruningModule):
    def __init__(self, mask=False):
        super(LeNet_5, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.conv1 = nn.Conv2d(1, 6, kernel_size=(5, 5))
        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.conv3 = nn.Conv2d(16, 120, kernel_size=(4,4))
        # self.conv4 = nn.Conv2d(16, 120, kernel_size=(5,5))
        self.fc1 = linear(120, 84)
        self.fc2 = linear(84, 10)

    def forward(self, x):
        # Conv1
        # print("# size x", x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        # print("#size conv1", x.shape )
        # Conv2
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, kernel_size=(2, 2), stride=2)
        # print("#size conv2", x.shape )


        # Conv3
        x = self.conv3(x)
        x = F.relu(x)
        # print("#size conv3", x.shape )


        # Fully-connected
        x = x.view(-1, 120)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.log_softmax(x, dim=1)

        return x

import torch.nn as nn
from collections import OrderedDict


class vgg19(PruningModule):
    """
    Input - 1x32x32
    C1 - 6@28x28 (5x5 kernel)
    tanh
    S2 - 6@14x14 (2x2 kernel, stride 2) Subsampling
    C3 - 16@10x10 (5x5 kernel, complicated shit)
    tanh
    S4 - 16@5x5 (2x2 kernel, stride 2) Subsampling
    C5 - 120@1x1 (5x5 kernel)
    F6 - 84
    tanh
    F7 - 10 (Output)
    """
    def __init__(self, mask=False):
        super(vgg19, self).__init__()
        linear = MaskedLinear if mask else nn.Linear
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 64, kernel_size=(3, 3), padding=1)),
            ('relu1', nn.ReLU()),
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c2', nn.Conv2d(64, 128, kernel_size=(3, 3), padding=1)),
            ('relu2', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c3', nn.Conv2d(128, 256, kernel_size=(3, 3), padding=1)),
            ('relu3', nn.ReLU()),
            ('c4', nn.Conv2d(256, 256, kernel_size=(3, 3), padding=1)),
            ('relu4', nn.ReLU()),
            ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c5', nn.Conv2d(256, 512, kernel_size=(3, 3), padding=1)),
            ('relu5', nn.ReLU()),
            ('c6', nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)),
            ('relu6', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

            ('c7', nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)),
            ('relu7', nn.ReLU()),
            ('c8', nn.Conv2d(512, 512, kernel_size=(3, 3), padding=1)),
            ('relu8', nn.ReLU()),
            ('s5', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),

        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('d1',nn.Dropout()),
            ('l1',linear(512, 512)),
            ('relu9',nn.ReLU(True)),
            ('d2',nn.Dropout()),
            ('l2',linear(512, 512)),
            ('relu10',nn.ReLU(True)),
            ('l3',linear(512, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output
