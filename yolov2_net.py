import torch
from torch.autograd import Variable
import torch.nn as NN
from torch.optim import Adam, SGD


class YOLOv2(NN.module):
    '''
    Darknet-19 architecture.
    '''

    def _initialize_weights(self):
        '''
        Weight initialization module.
        '''
        for m in self.modules():

            if isinstance(m, NN.Conv2d):
                NN.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    NN.init.constant_(m.bias, 0)
            elif isinstance(m, NN.BatchNorm2d):
                NN.init.constant_(m.weight, 1)
                NN.init.constant_(m.bias, 0)

    def __init__(self, k, num_classes, init_weights=True):

        super(YOLOv2, self).__init__()

        #configuration of the layers in terms of (kernel_size, filter_channel_output). 'M' stands for maxpooling.
        self.cfgs = {
            'yolo':[(3,32), 'M', (3,64), 'M', (3,128), (1,64), (3,128), 'M', (3,256), (1,128), (3,256), 'M', (3,512),
                    (1,256), (3,256), (1,256), (3,512), 'M', (3,1024), (1,512), (3,1024), (1,512), (3,1024), (3,1024), (3,1024), (3,1024), (3, k*(num_classes*5))]
        }

        layers = []
        in_channels = 3

        for l in self.cfgs['yolo']:

            if l == 'M':
                layers += [NN.MaxPool2d(kernel_size=2, stride=2)]
            else:
                conv2d = NN.Conv2d(in_channels, l[1], kernel_size=l[0], padding=1)

                #for the last convolution layer. No activation function.
                if l[1] == k*(num_classes*5):
                    layers += [conv2d, NN.BatchNorm2d(num_features=l[1])]
                    break

                layers += [conv2d, NN.BatchNorm2d(num_features=l[1]), NN.LeakyReLU(inplace=True)]
                in_channels = l[1]

        self.features = NN.Sequential(*layers)
        
        
        if init_weights:
            self._initialize_weights()
        
    
    def forward(self, input_x):

        x = self.feature(input_x)
        x = torch.flatten(x, 1)

        return x

        


            