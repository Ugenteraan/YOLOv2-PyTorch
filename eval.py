'''
Testing script for YOLO.
1) Read the file names from the input folder to a list.
2) Load the files using pyTorch dataloader.
3) Feed the dataset by batches into the evaluation model.
4) Use NMS on the output.
5) Draw the bounding box(es) on the images and write the modified images into the output folder.
'''
import torch
import numpy as np
from yolo_net import YOLO
from utils import create_test_lists
import cfg


SAVED_MODEL = torch.load(cfg.TRAINED_MODEL_PATH_FOLDER+cfg.TRAINED_MODEL_NAME)
YOLO.load_state_dict(SAVED_MODEL)
YOLO.eval()





