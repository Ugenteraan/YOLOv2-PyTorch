'''
Configuration file.
'''
import glob
import os
import torch
from utils import get_classes, imgnet_get_classes, imgnet_check_model


###IMAGENET config
IMGNET_DATASET_PATH = '../ImageNet'
IMGNET_CLASSES = imgnet_get_classes(folder_path=IMGNET_DATASET_PATH)
IMGNET_NUM_OF_CLASS = len(IMGNET_CLASSES)
IMGNET_MODEL_SAVE_PATH_FOLDER = './imagenet_model/'
IMGNET_MODEL_SAVE_NAME = 'imagenet_model.pth'
IMGNET_MODEL_PRESENCE = imgnet_check_model(model_path=IMGNET_MODEL_SAVE_PATH_FOLDER+IMGNET_MODEL_SAVE_NAME)
IMGNET_LEARNING_RATE = 1e-3
IMGNET_LEARNING_RATE_DECAY = 0.9
IMGNET_TOTAL_EPOCH = 160
IMGNET_BATCH_SIZE = 50
IMGNET_IMAGE_SIZE = 224
###

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
DATA_IMAGES_PATH = '../VOCdevkit/VOC2012/JPEGImages'
DATA_ANNOTATION_PATH = '../VOCdevkit/VOC2012/Annotations'
TRAINED_MODEL_PATH_FOLDER = './yolo_model/'
TRAINED_MODEL_NAME = 'yolo.pth'
IMAGE_SIZES = [320, 352, 384, 416, 448, 480, 512, 544, 576, 608]
IMAGE_DEPTH = 3
DETECTION_CONV_SIZE = 3
SUBSAMPLED_RATIO = 32
K = 5 #number of anchor box in a grid
LEARNING_RATE = 1e-5
LEARNING_RATE_DECAY = 0.999
LAMBDA_COORD = 5
LAMBDA_NOOBJ = 0.5
EPSILON_VALUE = 1e-8
TOTAL_EPOCH = 1000
MAP_TOPN = 5
MAP_IOU_THRESH = 0.5
CONFIDENCE_THRESH = 0.8
BATCH_SIZE = 20
NMS_IOU_THRESH = 0.7

#Get the image and annotation file paths
LIST_IMAGES = sorted([x for x in glob.glob(DATA_IMAGES_PATH + '/**')]) #length : 17125
LIST_ANNOTATIONS = sorted([x for x in glob.glob(DATA_ANNOTATION_PATH + '/**')]) #length : 17125
TOTAL_IMAGES = len(LIST_IMAGES)

#create the model saving directories if they don't exist.
if not os.path.exists(IMGNET_MODEL_SAVE_PATH_FOLDER):
    os.makedirs(IMGNET_MODEL_SAVE_PATH_FOLDER)

if not os.path.exists(TRAINED_MODEL_PATH_FOLDER):
    os.makedirs(TRAINED_MODEL_PATH_FOLDER)

CLASSES = get_classes(xml_files=LIST_ANNOTATIONS)
NUM_OF_CLASS = len(CLASSES)
EXCLUDED_CLASSES = [] #if you'd like to exclude certain classes for training.
