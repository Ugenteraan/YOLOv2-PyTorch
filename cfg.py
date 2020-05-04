import glob
from utils import get_classes, ImgNet_get_classes, ImgNet_check_model

###IMAGENET config
ImgNet_dataset_path = '../ImageNet'
ImgNet_classes = ImgNet_get_classes(folder_path=ImgNet_dataset_path)
ImgNet_num_of_class = len(ImgNet_classes)
ImgNet_model_save_path = './imagenet_model/imagenet.pth'
ImgNet_model_presence = ImgNet_check_model(model_path = ImgNet_model_save_path)
ImgNet_learning_rate = 1e-3
ImgNet_learning_rate_decay = 0.9
ImgNet_total_epoch = 160
###

data_images_path     = '../VOCdevkit/VOC2012/JPEGImages'
data_annotation_path = '../VOCdevkit/VOC2012/Annotations'
trained_model_path = './trained_model/'
image_sizes = [320,352,384,416,448,480,512,544,570,608]
image_depth  = 3
detection_conv_size = 3
subsampled_ratio = 32
k = 5 #number of anchor box in a grid
learning_rate = 1e-4
learning_rate_decay = 0.95
lambda_coord = 5
lambda_noobj = 0.5
epsilon_value = 1e-8
total_epoch = 100
mAP_topN = 5
mAP_iou_thresh = 0.2
confidence_thresh = 0.6
batch_size = 30

#Get the image and annotation file paths
list_images      = sorted([x for x in glob.glob(data_images_path + '/**')])     #length : 17125
list_annotations = sorted([x for x in glob.glob(data_annotation_path + '/**')]) #length : 17125
total_images = len(list_images)

classes = get_classes(xml_files=list_annotations)
num_of_class = len(classes)
print(classes)
excluded_classes = [] #if you'd like to exclude certain classes for training.

