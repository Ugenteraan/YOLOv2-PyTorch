import numpy as np 
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import cfg
from utils import cluster_bounding_boxes, generate_anchors, generate_training_data



class ToTensor:
    '''
    Transforms the images and labels from numpy array to tensors.
    '''
    def __init__(self):
        pass 

    def __call__(self, sample):

        image                   = sample['image']
        regression_objectness   = sample['regression_objectness']
        classification          = sample['classification']

        image = image.transpose((2,0,1)) #pytorch requires the channel to be in the 1st dimension of the tensor,
        
        return {'image':torch.from_numpy(image.astype('float32')),
                'regression_objectness': torch.from_numpy(regression_objectness),
                'classification': torch.from_numpy(classification)}



class Load_Dataset(Dataset):
    '''
    Wraps the loading of the dataset in PyTorch's DataLoader module.
    '''

    def __init__(self, resized_image_size, k=5, classes = cfg.classes, list_images = cfg.list_images, list_annotations = cfg.list_annotations, 
                 total_images = cfg.total_images, subsampled_ratio=cfg.subsampled_ratio, detection_conv_size=cfg.detection_conv_size, 
                                                                                            excluded_classes=cfg.excluded_classes, transform=None):

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
        self.excluded_classes   = excluded_classes
        self.transform          = transform

        #get the top-k anchor sizes using modifed K-Means clustering.
        self.anchor_sizes = cluster_bounding_boxes(k=self.k, total_images=self.total_images, resized_image_size=self.resized_image_size, 
                                                   list_annotations=cfg.list_annotations, classes=cfg.classes, excluded_classes=cfg.excluded_classes)

        self.anchors_list  = generate_anchors(anchor_sizes=self.anchor_sizes, detection_conv_size=self.detection_conv_size, 
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
        
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        image, regression_objectness_array, class_label_array = generate_training_data(data_index=idx, anchors_list=self.anchors_list, 
                                        xml_file_path=self.list_annotations[idx], classes=self.classes, resized_image_size=self.resized_image_size, 
                                    subsampled_ratio=self.subsampled_ratio, excluded_classes=self.excluded_classes, image_path=self.list_images[idx])

        sample = {'image':image,
                 'regression_objectness':regression_objectness_array,
                 'classification' : class_label_array}
        
        if self.transform:
            sample = self.transform(sample)
        
        return sample