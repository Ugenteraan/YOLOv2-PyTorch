from load_data import Load_Dataset, ToTensor, ImgNet_loadDataset
import cfg
import torch
from torch.utils.data import DataLoader
import cv2
from label_format import calculate_ground_truth
import numpy as np
from yolo_net import yolo, optimizer, loss, lr_decay #decay rate update
from tqdm import tqdm
from post_process import PostProcess
from random import randint
# from darknet19 import darknet19, ImgNet_optimizer, ImgNet_lr_decay, ImgNet_criterion
import itertools
import os

print(yolo)
chosen_image_index = 0
highest_map = 0

training_losses_list = []
training_mAPs_list = []



training_data = Load_Dataset(resized_image_size=320, transform=ToTensor())

dataloader = DataLoader(training_data, batch_size=3, shuffle=False, num_workers=4)
for epoch_idx in range(cfg.total_epoch):
    
    epoch_loss = 0
    training_loss = []
    

    
    chosen_image_size = 320
    feature_size = int(chosen_image_size/cfg.subsampled_ratio)
    
    

    postProcess_obj = PostProcess(box_num_per_grid=cfg.k, feature_size=feature_size, topN_pred=cfg.mAP_topN, anchors_list=training_data.anchors_list)
    

    for i, sample in tqdm(enumerate(dataloader)):
        # print(sample["image"].shape)
        # print(sample["label"].shape)
        if i == 0:
            pass 
        else:
            continue
        
        batch_x, batch_y = sample["image"].to(cfg.device), sample["label"].to(cfg.device)
        
        optimizer.zero_grad()
        
        #[batch size, feature map width, feature map height, number of anchors, 5 + number of classes]
        outputs = yolo(batch_x) #THE OUTPUTS ARE NOT YET GONE THROUGH THE ACTIVATION FUNCTIONS.
        
        total_loss = loss(predicted_array= outputs, label_array=batch_y)
        # mAP_object._collect(predicted_boxes=outputs.detach().cpu().numpy(), gt_boxes=batch_y.cpu().numpy())
        # mAP_object.non_max_suppression(predictions=outputs.detach().cpu().numpy())
        postProcess_obj.nms(predictions=outputs.detach().contiguous())
        training_loss.append(total_loss.item())
        total_loss.backward()
        optimizer.step()                
        img_ = np.asarray(np.transpose(batch_x.cpu().numpy()[0], (1,2,0)))
        img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
        
        calculated_batch = calculate_ground_truth(subsampled_ratio=32, anchors_list=training_data.anchors_list, resized_image_size=chosen_image_size, 
                            network_prediction=outputs.detach().cpu().numpy(), prob_threshold=0.7)
        
        gt_box = calculate_ground_truth(subsampled_ratio=32, anchors_list=training_data.anchors_list, resized_image_size=chosen_image_size, 
                            network_prediction=batch_y.cpu().numpy(), prob_threshold=0.9, ground_truth_mode=True)
        # print("CLASS : " , calculated_batch[0])
        
        for k in range(gt_box.shape[1]):
            # print(int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3]))
            # try:
            
            cv2.putText(img, (str(cfg.classes[int(gt_box[0][k][5])])), (int(gt_box[0][k][1])+20, 
                                                                                  int(gt_box[0][k][2])-10), cv2.FONT_HERSHEY_SIMPLEX, 
                                                                                                                                    0.4, (36,255,12), 2)
            cv2.rectangle(img, (int(gt_box[0][k][1]), int(gt_box[0][k][2])), (int(gt_box[0][k][3]), int(gt_box[0][k][4])),
                        (255,0,0), 1)
            # except Exception as e:
            #     print(e)
            #     pass
        for k in range(calculated_batch.shape[1]):
            # print(int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3]))
            try:
            
                cv2.putText(img, (str(round(calculated_batch[0][k][0],4))+", "+ str(cfg.classes[int(calculated_batch[0][k][5])])), (int(calculated_batch[0][k][1]), 
                                                                                    int(calculated_batch[0][k][2])-8), cv2.FONT_HERSHEY_SIMPLEX, 
                                                                                                                                        0.4, (36,255,12), 2)
                cv2.rectangle(img, (int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2])), (int(calculated_batch[0][k][3]), int(calculated_batch[0][k][4])),
                        (0,255,0), 1)
            except Exception as e:
                print(e)
                pass
            
        cv2.imshow("img", img)
        cv2.waitKey(0)
        break  
        
        
    
    lr_decay.step() #decay rate update

    # meanAP = mAP_object.calculate_meanAP()
    # print("MEAN Avg Prec : ", meanAP)
    training_loss = np.average(training_loss)
    print("Epoch %d, \t Loss : %g"%(epoch_idx, training_loss))
    
    training_losses_list.append(training_loss)
    # training_mAPs_list.append(meanAP)
    
    


ap_file = open("map.txt", 'w+')
ap_file.write(str(training_mAPs_list))
ap_file.close()

loss_file = open("loss.txt", "w+")
loss_file.write(str(training_losses_list))
loss_file.close()
    








    # img_ = np.asarray(np.transpose(batch_x.cpu().numpy()[0], (1,2,0)))
    # img = cv2.cvtColor(img_, cv2.COLOR_BGR2RGB)
    
    # calculated_batch = calculate_ground_truth(subsampled_ratio=32, anchors_list=training_data.anchors_list, resized_image_size=320, 
    #                     network_prediction=outputs.detach().cpu().numpy(), prob_threshold=0.99)
    
    
    # # print("CLASS : " , calculated_batch[0])
    # for k in range(calculated_batch.shape[1]):
    #     # print(int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3]))
    #     # try:
        
    #     cv2.putText(img, (str(round(calculated_batch[0][k][0],4))+", "+ str(cfg.classes[int(calculated_batch[0][k][5])])), (int(calculated_batch[0][k][1]), 
    #                                                                           int(calculated_batch[0][k][2])-8), cv2.FONT_HERSHEY_SIMPLEX, 
    #                                                                                                                             0.4, (36,255,12), 2)
    #     cv2.rectangle(img, (int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2])), (int(calculated_batch[0][k][3]), int(calculated_batch[0][k][4])),
    #                 (0,255,0), 1)
    #     # except Exception as e:
    #     #     print(e)
    #     #     pass
        
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
    # break  
    
    