'''
Loads the required dataset for both ImageNet model training and YOLOv2 model training.
'''
import torch
from torch.utils.data import Dataset
import cfg
from utils import cluster_bounding_boxes, generate_anchors, generate_training_data, imgnet_generate_data, imgnet_read_data



class ToTensor:
    '''
    Transforms the images and labels from numpy array to tensors.
    '''
    def __init__(self):
        pass

    def __call__(self, sample):

        image = sample['image']
        label = sample['label']

        image = image.transpose((2, 0, 1)) #pytorch requires the channel to be in the 1st dimension of the tensor,


        return {'image':torch.from_numpy(image.astype('float32')),
                'label': torch.from_numpy(label)}



class LoadDataset(Dataset):
    '''
    Wraps the loading of the dataset in PyTorch's DataLoader module.
    '''

    def __init__(self, resized_image_size, k=cfg.K, classes=cfg.CLASSES.copy(), list_images=cfg.LIST_IMAGES, list_annotations=cfg.LIST_ANNOTATIONS,
                 total_images=cfg.TOTAL_IMAGES, subsampled_ratio=cfg.SUBSAMPLED_RATIO, detection_conv_size=cfg.DETECTION_CONV_SIZE,
                 excluded_classes=cfg.EXCLUDED_CLASSES.copy(), transform=None):

        '''
        Initialize parameters and anchors using KMeans.
        '''

        self.resized_image_size = resized_image_size
        self.classes = classes
        self.list_images = list_images
        self.list_annotations = list_annotations
        self.total_images = total_images
        self.k = k
        self.subsampled_ratio = subsampled_ratio
        self.detection_conv_size = detection_conv_size
        self.excluded_classes = excluded_classes
        self.transform = transform

        #get the top-k anchor sizes using modifed K-Means clustering.
        self.anchor_sizes = cluster_bounding_boxes(k=self.k, total_images=self.total_images, resized_image_size=self.resized_image_size,
                                                   list_annotations=cfg.LIST_ANNOTATIONS, classes=cfg.CLASSES, excluded_classes=cfg.EXCLUDED_CLASSES)

        self.anchors_list = generate_anchors(anchor_sizes=self.anchor_sizes,
                                             subsampled_ratio=self.subsampled_ratio, resized_image_size=self.resized_image_size)



    def __len__(self):
        '''
        Abstract method. Returns the total number of data.
        '''
        return self.total_images


    def __getitem__(self, idx):
        '''
        Abstract method. Returns the image and label for a single input with the index of `idx`.
        '''

        if torch.is_tensor(idx):
            idx = idx.tolist()


        image, label_array = generate_training_data(anchors_list=self.anchors_list,
                                        xml_file_path=self.list_annotations[idx], classes=self.classes, resized_image_size=self.resized_image_size,
                                    subsampled_ratio=self.subsampled_ratio, excluded_classes=self.excluded_classes, image_path=self.list_images[idx])

        sample = {'image':image,
                 'label':label_array}

        if self.transform:
            sample = self.transform(sample)

        return sample


class ImgnetLoadDataset(Dataset):
    '''
     Wraps the loading of the ImageNet dataset in PyTorch's DataLoader module.
    '''

    def __init__(self, resized_image_size, class_list, dataset_folder_path, transform=None):
        '''
        Initialize parameters and generate the images path list and corresponding labels.
        '''

        self.resized_image_size = resized_image_size
        self.class_list = class_list
        self.dataset_folder_path = dataset_folder_path
        self.transform = transform

        self.images_path_list, self.labels_list = imgnet_generate_data(folder_path=self.dataset_folder_path,
                                                                      class_list=self.class_list)


    def __len__(self):
        '''
        Returns the number of data in the list.
        '''

        return len(self.images_path_list)

    def __getitem__(self, idx):
        '''
        Read one single data.
        '''
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image_array, label_array = imgnet_read_data(image_path=self.images_path_list[idx], class_idx=self.labels_list[idx],
                                                    resized_image_size=self.resized_image_size)

        sample = {
            'image':image_array,
            'label':label_array
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
