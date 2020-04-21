from load_data import Load_Dataset
import cfg
from torch.utils.data import DataLoader
import cv2


training_data = Load_Dataset(resized_image_size=320)

dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=1)

for i, sample in enumerate(dataloader):
    print(sample["image"].shape)
    print(sample["regression_objectness"].shape)