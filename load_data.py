import numpy as np 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cfg
from utils import cluster_bounding_boxes, generate_anchors


class Load_Dataset(Dataset):
    '''
    Wraps the loading of the dataset in PyTorch's DataLoader module.
    '''

    def __init__(self, resized_image_size, k=5, classes = cfg.classes, list_images = cfg.list_images, list_annotations = cfg.list_annotations, 
                 total_images = cfg.total_images, subsampled_ratio=cfg.subsampled_ratio, detection_conv_size=cfg.detection_conv_size):

        '''
        Initialize parameters and anchors using KMeans.
        '''

        self.resized_image_size = resized_image_size
        self.classes            = classes 
        self.list_images        = list_images
        self.list_annotations   = list_annotations
        self.total_images       = total_images
        self.k                  = k
        self.subsampled_ratio   = subsampled_ratio
        self.detection_conv_size= detection_conv_size

        #get the top-k anchor sizes using modifed K-Means clustering.
        self.anchor_sizes = cluster_bounding_boxes(k=self.k, total_images=self.total_images, resized_image_size=self.resized_image_size, 
                                                   list_annotations=cfg.list_annotations, classes=cfg.classes, excluded_classes=cfg.excluded_classes)

        self.anchor_list  = generate_anchors(anchor_sizes=self.anchor_sizes, detection_conv_size=self.detection_conv_size, 
                                             subsampled_ratio=self.subsampled_ratio, resized_image_size=self.resized_image_size)
        

    
    def __len__(self):
        '''
        Abstract method. Returns the total number of data.
        '''
        return self.total_images

    
    def __getitem__(self, idx):
        '''
        Abstract method. Returns the label for a single input with the index of `idx`.
        '''
    
        

    
