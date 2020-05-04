from load_data import Load_Dataset, ToTensor, ImgNet_loadDataset
import cfg
import torch
from torch.utils.data import DataLoader
import cv2
from label_format import calculate_ground_truth
import numpy as np
# from yolo_net import yolo, optimizer, loss, lr_decay #decay rate update
from tqdm import tqdm
from mAP import mAP
from random import randint
from darknet19 import darknet19, ImgNet_optimizer, ImgNet_lr_decay, ImgNet_criterion



if not cfg.ImgNet_model_presence:
    '''
    If the classification model is not present, then we'll have to train the model with the ImageNet images for classification.
    '''
    print(darknet19)
    
    ImgNet_training_data = ImgNet_loadDataset(resized_image_size=224, class_list=cfg.ImgNet_classes, dataset_folder_path=cfg.ImgNet_dataset_path,
                                       transform=ToTensor())
    
    ImgNet_dataloader = DataLoader(ImgNet_training_data, batch_size=cfg.ImgNet_batch_size, shuffle=True, num_workers=4)
    
    best_accuracy = 0
    for epoch_idx in range(cfg.ImgNet_total_epoch):
        
        epoch_training_loss = []
        epoch_accuracy = []
    
        for i, sample in tqdm(enumerate(ImgNet_dataloader)):
            
            batch_x, batch_y = sample["image"].cuda(), sample["label"].cuda()
            
            ImgNet_optimizer.zero_grad()
            
            classification_output = darknet19(batch_x)

            training_loss = ImgNet_criterion(input=classification_output, target=batch_y)
            
            epoch_training_loss.append(training_loss)
            
            training_loss.backward()
            ImgNet_optimizer.step()
            
            batch_acc = darknet19.calculate_accuracy(network_output=classification_output, target=batch_y)
            epoch_accuracy.append(batch_acc)
            
        
        ImgNet_lr_decay.step()
        
        current_accuracy = np.average(epoch_accuracy)
        print("Epoch %d, \t Training Loss : %g, \t Training Accuracy : %g"%(epoch_idx, np.average(training_loss), current_accuracy))
        
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            torch.save(darknet19.state_dict(), cfg.ImgNet_model_save_path)

            

'''
print(yolo)
chosen_image_index = 0
highest_map = 0

for epoch_idx in range(cfg.total_epoch):
    
    epoch_loss = 0
    training_loss = []
    
    
    if epoch_idx % 10 == 0:
        #there are 10 options for image sizes.
        chosen_image_index = randint(0,9)
    
    chosen_image_size = cfg.image_sizes[chosen_image_index]
    feature_size = int(chosen_image_size/cfg.subsampled_ratio)
    
    training_data = Load_Dataset(resized_image_size=chosen_image_size, transform=ToTensor())

    dataloader = DataLoader(training_data, batch_size=cfg.batch_size, shuffle=True, num_workers=4)

    mAP_object = mAP(box_num_per_grid=cfg.k, feature_size=feature_size, topN_pred=cfg.mAP_topN, anchors_list=training_data.anchors_list)

    for i, sample in tqdm(enumerate(dataloader)):
        # print(sample["image"].shape)
        # print(sample["label"].shape)
        
        batch_x, batch_y = sample["image"].cuda(), sample["label"].cuda()
        
        optimizer.zero_grad()
        
        #[batch size, feature map width, feature map height, number of anchors, 5 + number of classes]
        outputs = yolo(batch_x) #THE OUTPUTS ARE NOT YET GONE THROUGH THE ACTIVATION FUNCTIONS.
        
        total_loss = loss(predicted_array= outputs, label_array=batch_y)
        mAP_object._collect(predicted_boxes=outputs.detach().cpu().numpy(), gt_boxes=batch_y.cpu().numpy())
        training_loss.append(total_loss.item())
        total_loss.backward()
        optimizer.step()                 
        
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
        
    
    lr_decay.step() #decay rate update
    # print(total_loss.item())
    meanAP = mAP_object.calculate_meanAP()
    print("MEAN Avg Prec : ", meanAP)
    training_loss = np.average(training_loss)
    print("Epoch %d, \t Loss : %g"%(epoch_idx, training_loss))
    
    if meanAP > highest_map:
        highest_map = meanAP
        torch.save(yolo.state_dict(), './yolo_model.pth')
        
    
    

    # img = sample["image"].numpy()[0]
    # print(calculated_batch.shape)
    # calculated_batch = calculate_ground_truth(subsampled_ratio=32, anchors_list=training_data.anchors_list, resized_image_size=320, 
                            # network_prediction=batch_y.cpu().numpy(), prob_threshold=0.85)
    # for k in range(calculated_batch.shape[1]):
    #     print(int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1]), int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3]))
    #     cv2.rectangle(img, (int(calculated_batch[0][k][0]), int(calculated_batch[0][k][1])), (int(calculated_batch[0][k][2]), int(calculated_batch[0][k][3])),
    #                     (255,255,255), 2)
    
    # cv2.imshow("img", img)
    # cv2.waitKey(0)
'''