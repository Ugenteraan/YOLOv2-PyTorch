from load_data import Load_Dataset
import cfg
from torch.utils.data import DataLoader
import cv2
from label_format import calculate_ground_truth
import numpy as np


training_data = Load_Dataset(resized_image_size=320)

dataloader = DataLoader(training_data, batch_size=1, shuffle=True, num_workers=4)

for i, sample in enumerate(dataloader):
    print(sample["image"].shape)
    print(sample["regression_objectness"].shape)

    calculated_batch = calculate_ground_truth(subsampled_ratio=32, anchors_list=training_data.anchors_list, resized_image_size=320, 
                            network_prediction=sample["regression_objectness"].numpy(), prob_threshold=0.5 )

    img = sample["image"].numpy()[0]
    print(calculated_batch.shape)
    for k in range(calculated_batch.shape[1]):
        print(int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3]))
        cv2.rectangle(img, (int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1])), (int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3])),
                        (255,255,255), 2)
    
    cv2.imshow("img", img)
    cv2.waitKey(0)