import torch
import torch.nn as NN
from torch.optim import Adam
import cfg

class YOLOv2(NN.Module):
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
                    NN.init.constant_(m.bias, 0.5)
            elif isinstance(m, NN.BatchNorm2d):
                NN.init.constant_(m.weight, 1)
                NN.init.constant_(m.bias, 0.5)

    def __init__(self, k, num_classes, init_weights=True):
        
        self.k = k
        self.num_classes = num_classes

        super(YOLOv2, self).__init__()

        #configuration of the layers in terms of (kernel_size, filter_channel_output). 'M' stands for maxpooling.
        self.cfgs = {
            'yolo':[(3,32), 'M', (3,64), 'M', (3,128), (1,64), (3,128), 'M', (3,256), (1,128), (3,256), 'M', (3,512),
                    (1,256), (3,256), (1,256), (3,512), 'M', (3,1024), (1,512), (3,1024), (1,512), (3,1024), (3,1024), (3,1024), (3,1024), 
                    (cfg.detection_conv_size, self.k*(self.num_classes+5))]
        }

        layers = []
        in_channels = 3
        

        for l in self.cfgs['yolo']:
            
            padding_value = 1
            
            if l == 'M':
                layers += [NN.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if l[0] == 1: #if the filter size is 1, no padding is required. Else, the input dimension will increase by 1.
                    padding_value = 0
                conv2d = NN.Conv2d(in_channels, l[1], kernel_size=l[0], padding=padding_value)

                #for the last convolution layer. No activation function.
                if l[1] == k*(num_classes+5):
                    layers += [conv2d, NN.BatchNorm2d(num_features=l[1])]
                    break

                layers += [conv2d, NN.BatchNorm2d(num_features=l[1]), NN.LeakyReLU(inplace=True)]
                in_channels = l[1]

        self.features = NN.Sequential(*layers)
        
         
        if init_weights:
            self._initialize_weights()
        
    
    def forward(self, input_x):

        x = self.features(input_x)
        
        subsampled_feature_size = x.size()[-1] #get the size of the feature map
        
        x = torch.transpose(x, 1, -1) 
        #reshape the output into [batch size, feature map width, feature map height, number of anchors, 5 + number of classes]
        x = x.view(-1, subsampled_feature_size, subsampled_feature_size, self.k, 5+self.num_classes)
        
        return x
    

    
def loss(predicted_array, label_array):
    '''
    Loss function for Yolov2.
    '''
    
    #information on which anchor contains object and which anchor don't.
    gt_objectness = label_array[:,:,:,:,0:1]
    
    mask = torch.ones_like(gt_objectness)
    
    gt_noObjectness = mask - gt_objectness
    
    #get the center values from both the arrays.
    predicted_centerX = torch.nn.Sigmoid()(predicted_array[:,:,:,:,1:2])
    predicted_centerY = torch.nn.Sigmoid()(predicted_array[:,:,:,:,2:3])
    gt_centerX        = label_array[:,:,:,:,1:2]
    gt_centerY        = label_array[:,:,:,:,2:3]
    
    center_loss = cfg.lambda_coord * torch.sum(gt_objectness * ((predicted_centerX - gt_centerX)**2 + (gt_centerY - predicted_centerY)**2))
    
    #get the height and width values from both the arrays.
    predicted_width     = predicted_array[:,:,:,:,3:4]
    predicted_height    = predicted_array[:,:,:,:,4:5]
    gt_width            = label_array[:,:,:,:,3:4]
    gt_height           = label_array[:,:,:,:,4:5]
    
    #actual size loss according to YOLO 1 loss function. But I don't think square root would work since the ground truth values
    #can have negative values.
    # size_loss = cfg.lambda_coord * gt_objectness * ((torch.sqrt(predicted_width + cfg.epsilon_value) - torch.sqrt(gt_width)**2 + 
    #                                                 (torch.sqrt(predicted_height + cfg.epsilon_value) - torch.sqrt(gt_height)**2)
    
    size_loss = cfg.lambda_coord * torch.sum(gt_objectness * ((predicted_width  - gt_width)**2 + (predicted_height - gt_height)**2))
    
    #get the predicted probability of objectness
    predicted_objectness = torch.nn.Sigmoid()(predicted_array[:,:,:,:,0:1])
    
    objectness_loss = torch.sum(gt_objectness*(gt_objectness - predicted_objectness)**2)
    
    wrong_objectness_loss = cfg.lambda_noobj * torch.sum(gt_noObjectness*(gt_objectness - predicted_objectness)**2)
    
    #get the predicted probability of the classes and the ground truth class probabilities.
    predicted_classes = predicted_array[:,:,:,:,5:]
    gt_classes        = label_array[:,:,:,:,5].type(torch.int64)
    
    # print(gt_objectness.size())
    # print(gt_classes.size())
    # print(predicted_classes.size())
    # print((gt_objectness*predicted_classes).size())
    
    #cross-entropy loss between the predicted and label
    # classification_loss = NN.CrossEntropyLoss()(gt_classes, gt_objectness*predicted_classes)
    
    #sum all the losses together
    total_loss = center_loss + size_loss + objectness_loss + wrong_objectness_loss 
    
    return total_loss
    
    

    
yolo = YOLOv2(k=cfg.k, num_classes=cfg.num_of_class, init_weights=True)

optimizer = Adam(yolo.parameters(), lr = cfg.learning_rate)


if torch.cuda.is_available():
    
    yolo = yolo.cuda()





        


            