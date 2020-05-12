import glob
import os
from utils import get_classes, ImgNet_get_classes, ImgNet_check_model

###IMAGENET config
ImgNet_dataset_path = '../ImageNet'
ImgNet_classes = ImgNet_get_classes(folder_path=ImgNet_dataset_path)
ImgNet_num_of_class = len(ImgNet_classes)
ImgNet_model_save_path_folder = './imagenet_model/'
ImgNet_model_save_name = 'imagenet_model.pth'
ImgNet_model_presence = ImgNet_check_model(model_path = ImgNet_model_save_path_folder + ImgNet_model_save_name)
ImgNet_learning_rate = 1e-3
ImgNet_learning_rate_decay = 0.9
ImgNet_total_epoch = 160
ImgNet_batch_size = 50
ImgNet_image_size = 224
###

data_images_path     = '../VOCdevkit/VOC2012/JPEGImages'
data_annotation_path = '../VOCdevkit/VOC2012/Annotations'
trained_model_path_folder = './yolo_model/'
trained_model_name  = 'yolo.pth'
image_sizes = [320,352,384,416,448,480,512,544,576,608]
image_depth  = 3
detection_conv_size = 3
subsampled_ratio = 32
k = 5 #number of anchor box in a grid
learning_rate = 1e-3
learning_rate_decay = 0.96
lambda_coord = 20
lambda_noobj = 0.5
epsilon_value = 1e-8
total_epoch = 1000
mAP_topN = 5
mAP_iou_thresh = 0.5
confidence_thresh = 0.7
batch_size = 20

#Get the image and annotation file paths
list_images      = sorted([x for x in glob.glob(data_images_path + '/**')])     #length : 17125
list_annotations = sorted([x for x in glob.glob(data_annotation_path + '/**')]) #length : 17125
total_images = len(list_images)

#create the model saving directories if they don't exist.
if not os.path.exists(ImgNet_model_save_path_folder):
    os.makedirs(ImgNet_model_save_path_folder)

if not os.path.exists(trained_model_path_folder):
    os.makedirs(trained_model_path_folder)

classes = get_classes(xml_files=list_annotations)
num_of_class = len(classes)
excluded_classes = [] #if you'd like to exclude certain classes for training.

