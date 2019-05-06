import torch
import torch.nn as nn
from collections import OrderedDict

class Fire(nn.Module):
##
## Another way of Representing fire module which intakes number of input layers
## and the multiplier, the factor by which we want to increase the number of
## output layers

## e.g. : ('fire1', Fire(10,2)) -> num. InputLayers = 10, num. OutputLayers = 20

    # def __init__(self, base_layer_dim,decision):
    #     super(Fire, self).__init__()

    #     P_expand_3x3 = 0.5
    #     squeeze_ratio = 0.25
    #     nExpand = int(base_layer_dim * decision)
    #     expand3x3_planes = int(nExpand * P_expand_3x3)
    #     expand1x1_planes = int(nExpand * (1 - P_expand_3x3))
    #     squeeze1x1_planes = int(squeeze_ratio * nExpand)

    #     self.squeeze = nn.Conv2d(base_layer_dim, squeeze1x1_planes, kernel_size=1)
    #     self.squeeze_activation = nn.ReLU(inplace=True)
        
    #     self.expand1x1 = nn.Conv2d(squeeze1x1_planes, expand1x1_planes,
    #                                kernel_size=1)
    #     self.expand1x1_activation = nn.ReLU(inplace=True)
    #     self.expand3x3 = nn.Conv2d(squeeze1x1_planes, expand3x3_planes,
    #                                kernel_size=3, padding=1)
    #     self.expand3x3_activation = nn.ReLU(inplace=True)
##
    def __init__(self, inplanes, squeeze_planes,
                 expand1x1_planes, expand3x3_planes):
        super(Fire, self).__init__()
        
        self.inplanes = inplanes
        self.squeeze = nn.Conv2d(inplanes, squeeze_planes, kernel_size=1)
        self.squeeze_activation = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(squeeze_planes, expand1x1_planes,
                                   kernel_size=1)
        self.expand1x1_activation = nn.ReLU(inplace=True)
        self.expand3x3 = nn.Conv2d(squeeze_planes, expand3x3_planes,
                                   kernel_size=3, padding=1)
        self.expand3x3_activation = nn.ReLU(inplace=True)
        # self.expand5x5 = nn.Conv2d(squeeze_planes, expand5x5_planes,
        #                            kernel_size=5, padding=2)
        # self.expand5x5_activation = nn.ReLU(inplace=True)        

    def forward(self, x):
        x = self.squeeze_activation(self.squeeze(x))
        return torch.cat([
            self.expand1x1_activation(self.expand1x1(x)),
            self.expand3x3_activation(self.expand3x3(x)),
            # self.expand5x5_activation(self.expand5x5(x))
        ], 1)

class LeNet5(nn.Module):
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
    def __init__(self):
        # super(LeNet5, self).__init__()

#       freq              = 2
#       base_layer_dim    = 128
#       squeeze_ratio     = 0.125
#       P_expand_3x3      = 0.5
#       inc_nlayers       = 128



#########accuracy: 98...72kB########
        super(LeNet5, self).__init__()
        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 5, kernel_size=(3, 3))),
            # ('relu1', nn.ReLU()),
            ('c3', nn.Conv2d(5, 10, kernel_size=(3, 3))),
            ('relu1', nn.ReLU()),  
            ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire1', Fire(10, 5, 10, 10)),
            #('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire3', Fire(20, 10, 20, 20)),
            ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('fire4', Fire(40,10, 20, 20)),
            ('fire5', Fire(40,5, 10, 10)),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=1)),
            ('fire6', Fire(20, 5, 10, 10)),
            ('fire7', Fire(20, 5, 10, 10)),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c2', nn.Conv2d(20, 10, kernel_size=(3, 3)))

        ]))


###########accuracy-98.44 | 98.7  , 92kb|93kb  ######

        # self.convnet = nn.Sequential(OrderedDict([
        #     ('c1', nn.Conv2d(1, 10, kernel_size=(3, 3))),
        #     # ('relu1', nn.ReLU()),
        #     ('c3', nn.Conv2d(10, 20, kernel_size=(3, 3))),
        #     ('relu1', nn.ReLU()),  
        #     ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire1', Fire(20, 5, 10, 10)),
        #     #('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire3', Fire(20, 10, 20, 20)),
        #     ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire4', Fire(40,10, 20, 20)),
        #     ('fire5', Fire(40,10, 20, 20)),
        #     ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=1)),
        #     ('fire6', Fire(40, 10, 20, 20)),
        #     ('fire7', Fire(40, 5, 10, 10)),
        #     ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('c2', nn.Conv2d(20, 10, kernel_size=(3, 3)))

        # ]))
########################### 98.43 85kb model_size ###############
        # self.convnet = nn.Sequential(OrderedDict([
        #     ('c1', nn.Conv2d(1, 20, kernel_size=(5, 5))),
        #     ('relu1', nn.ReLU()), 
        #     ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire1', Fire(20, 5, 10, 10)),
        #     #('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire3', Fire(20, 10, 20, 20)),
        #     ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire4', Fire(40,10, 20, 20)),
        #     ('fire5', Fire(40,10, 20, 20)),
        #     ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=1)),
        #     ('fire6', Fire(40, 10, 20, 20)),
        #     ('fire7', Fire(40, 5, 10, 10)),
        #     ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('c2', nn.Conv2d(20, 10, kernel_size=(3, 3)))

        # ]))
################## accuracy 98.5 134 kb ################################
        # self.convnet = nn.Sequential(OrderedDict([
        #     ('c1', nn.Conv2d(1, 20, kernel_size=(5, 5))),
        #     ('relu1', nn.ReLU()), 
        #     ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire1', Fire(20, 5, 10, 10)),
        #     #('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire3', Fire(20, 10, 20, 20)),
        #     ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('fire4', Fire(40, 20, 40, 40)),
        #     ('fire5', Fire(80, 20, 40, 40)),
        #     ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=1)),
        #     ('fire6', Fire(80, 10, 20, 20)),
        #     ('fire7', Fire(40, 5, 10, 10)),
        #     ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #     ('c2', nn.Conv2d(20, 10, kernel_size=(3, 3))),
        #     # ('a1'   , nn.AvgPool2d(3,stride=1))
        #    # ('c3', nn.Conv2d(40, 10, kernel_size=(7, 7)))

        # ]))

        self.loss = nn.Sequential(OrderedDict([
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        # print(output.shape)

        output = output.view(img.size(0), -1)
        # print(output.shape)
        output = self.loss(output)
        return output

