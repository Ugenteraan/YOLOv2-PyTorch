import cfg
import numpy as np
import math



class mAP:
    '''

    1) Rank the predicted boxes in decreasing order based on the confidence score and choose top-N boxes.
    2) For every chosen boxes, determine if they are TP, FP or FN based on its IoU with the ground-truth boxes (There is no TN since an image 
    contains at least 1 object).
        a) If the chosen box has more than a certain set threshold with a ground-truth box and the class is also predicted correctly, it's a TP.
        b) If the chosen box has more than a certain set threshold with a ground-truth box but the class is predicted wrongly, it's a FN.
        c) If a ground-truth box is missed without a predicted box, then it's a FN. (FN is not needed to calculate Interpolated Precision and/or 
           Average Precision)
        d) If the chosen box has less than a certain set threshold with a ground-truth box but class predicted correctly, it's a FP.
        e) If the chosen box has more than a certain set threshold with a ground-truth box, class is predicted correctly, but it's a duplicated
        prediction, then it's a FP.
    3) Calculate the recall and precision based on the TP, FP and FN for each class.
    4) Calculate the Interpolated Precision (Calculated at each recall level) for each class.
    5) Calculate Average Precision (AP) by taking the Area Under Curve of Interpolated Precision for each class.
    6) Mean the AP over all classes.
    NOTE : Precision = TP/all detections ..... Recall = TP/all ground truths
    '''
    
    def __init__(self, box_num_per_grid, feature_size, topN_pred, anchors_list, IoU_thresh=cfg.mAP_iou_thresh, confidence_thresh=cfg.confidence_thresh, 
                                                                                    subsampled_ratio=cfg.subsampled_ratio, num_class=cfg.num_of_class):
        '''
        Initialize parameters. 
        box_num_per_grid : the number of anchors/boxes in a grid/cell.
        topN_pred : the top N prediction for mAP calculation.
        anchors_list : generated list of anchors used for the retrieval of the bounding box coordinates.
        '''
        
        self.num_of_anchor      = box_num_per_grid
        self.top_N              = topN_pred
        self.anchors_list       = anchors_list
        self.feature_size       = feature_size
        self.subsampled_ratio   = subsampled_ratio
        self.iou_thresh         = IoU_thresh
        self.confidence_thresh  = confidence_thresh
        self.epoch_predboxes    = []
        self.epoch_gtBoxes      = []
        self.num_class          = num_class
        
    
    def _collect(self, predicted_boxes, gt_boxes):
        '''
        For every batch, we'll transform the regression values into coordinates and append the info into our lists in order to perform the mAP
        calculation at the end of an epoch.
        '''
        #get the top N predictions along with their indexes.
        top_boxes_regressionValue, top_indexes = self.get_topN_prediction(predicted_boxes)
        
        #transform the ground truth and the predicted regression values into coordinates to calculate mAP.
        transformed_top_pred_boxes = np.asarray(self.get_box_coordinates(boxes=top_boxes_regressionValue, indexes=top_indexes), dtype=np.float32)
        transformed_top_gt_boxes = self.get_box_coordinates(boxes=gt_boxes)
        
        #append the prediction box(es) and the gt box(es) for each image into the lists.
        for box in transformed_top_gt_boxes:
            self.epoch_gtBoxes.append(box)
        
        for box in transformed_top_pred_boxes:
            self.epoch_predboxes.append(box)
            
        return None
        
    
    
    def calculate_meanAP(self):
        '''
        Calculate the mean Avg. Precision at the end of every epoch using the collected data.
        1) Iterate through every image predictions and gt boxes and identify the class that we're calculating the AP for.
        2) For each of those prediction array, compare the IoU to their corresponding ground truth box(es) and select the largest IoU predicted box.
            a) If there are no GT box at all, the predictions are FPs
            b) If the largest IoU predicted box has more than the threshold IoU, then it's a TP and assign the GT box with 1.
            c) 

        '''
        
        
        AP = []
        for class_index in range(self.num_class):
            
            
            
            precision_list, recall_list = [], []
            
            true_positive  = 0
            false_positive = 0
            det_counter = 0
            gt_counter = 0 
            
            
            #iterate through prediction and gt boxes for every image.
            for each_image_pred_boxes, each_image_gt_boxes in zip(self.epoch_predboxes, self.epoch_gtBoxes):
                
                #holds the prediction boxes and the gt boxes that belongs to a certain class.
                classed_pred_boxes, classed_gt_boxes = [], []
                
                for pred_box in each_image_pred_boxes: #iterate through every prediction box for an image.
                    
                    #check if the pred box predicted the class we're calculating the AP for.
                    if int(pred_box[5]) == class_index and pred_box[0] > self.confidence_thresh:
                        
                        classed_pred_boxes.append(pred_box)
                        det_counter += 1
                
                for gt_box in each_image_gt_boxes: #iterate through every ground-truth box for an image.
                    
                    #check if the gt box predicted belongs to the class we're calculating the AP for.
                    if int(gt_box[5]) == class_index:
                        
                        classed_gt_boxes.append(gt_box)
                        gt_counter += 1
                
                #if there are no ground truth box for such class but there are predictions for the class, all the predictions are FP.
                if (len(classed_gt_boxes) == 0) and (len(classed_pred_boxes) != 0):
                    
                    false_positive += len(classed_pred_boxes)
   
                    precision = true_positive/det_counter 
                    
                    try:
                        recall = true_positive/gt_counter
                    except ZeroDivisionError:
                        recall = 0 
                    
                    precision_list.append(precision)
                    recall_list.append(recall)      
                    
                    continue              
                
                #if there are no predictions and ground truth boxes for this class for this particular image.
                if (len(classed_gt_boxes) == 0) and (len(classed_pred_boxes) == 0):
                    
                    continue
                
                #for each predicted box, check the IoU with the ground-truth boxes and insert the index and the iou of ground-truth box that overlaps the 
                #most with the predicted box into the dictionary.
                IoU_mark = {}                 
                
                for index, each_pred_box in enumerate(classed_pred_boxes):
                    
                    #Predictions that has confidence lower than the threshold will not be contributing to the calculation of mAP.
                    #NOTE that the number of prediction boxes can now vary from 0 to "top_N"
                    if each_pred_box[0] < self.confidence_thresh:
                        continue
                    
                    #get the IoU between the predicted box and ALL the ground truth boxes in an image that belongs to this class index.
                    iou_array = self.calculate_iou(predicted_box = np.asarray(each_pred_box, dtype=np.float32), 
                                                   gt_boxes = np.asarray(classed_gt_boxes, dtype=np.float32))
                    
                    highest_iou_index = np.argmax(iou_array) #get the index of the gt box that has the highest IoU with the selected pred box.
                    
                    
                    try :
                        _ = IoU_mark[highest_iou_index]
                        
                    except KeyError:
                        
                        IoU_mark[highest_iou_index] = []
                    
                    #append the [index of pred box, IoU with the gt box]
                    IoU_mark[highest_iou_index].append([index, iou_array[highest_iou_index]])
                
                #sort the prediction boxes that belongs to a gt box in decreasing order.
                total_pred_boxes = 0 # to keep track of the prediction boxes that passed the confidence threshold above.
                TP_anchor_boxes = []
                for key in IoU_mark:
                    
                    #convert the list in the value to np array
                    original_array = np.asarray(IoU_mark[key], dtype=np.float32)

                    #sort the array in ascending order and get the last element for the highest IoU
                    highest_pred_elem = original_array[np.argsort(original_array[:,1])][-1]
                    
                    classed_gt_boxes[key][-1] = 1.0 #mark the last element of the gt box with 1 to denote assigned.
                    
                    TP_anchor_boxes.append(highest_pred_elem) #append each of the prediction boxes with the highest IoU to a particular Gt box.
                    
                    total_pred_boxes += len(IoU_mark[key]) #total up the number of prediction boxes.
                    
                #iterate through the predicted boxes to determine TP and FP and calculate precision and recall.
                for i in range(total_pred_boxes):
                    
                    #if the prediction box is not a TP
                    #these are duplicate prediction boxes.
                    if not i in [j[0] for j in TP_anchor_boxes]:
                        
                        false_positive += 1
                        
                        precision = true_positive/det_counter
                         
                        try:
                            recall = true_positive/gt_counter
                        except ZeroDivisionError:
                            recall = 0
                        
                        precision_list.append(precision)
                        recall_list.append(recall)
                    
                    else: #if the prediction is a TP
                        
                        true_positive += 1
                        
                        precision = true_positive/det_counter 
                        
                        try:
                            recall = true_positive/gt_counter
                        except ZeroDivisionError:
                            recall = 0
                        
                        precision_list.append(precision)
                        recall_list.append(recall)
                        
            #Calculate IP.
            #The array is reversed before we accumulate in order to go from high value to low value.
            #If the array is not reversed, then if the first detection is correct, i.e. precision = 1, then the rest of the array will be 1.
            reversed_precision      = np.asarray(precision_list[::-1])
            interpolated_precision  = np.flip(np.maximum.accumulate(reversed_precision))
            
            #Calculate AUC.
            Avg_precision = 0
            
            #-1 so that the "next_recall" doesn't go out of index range.
            for recall_index in range(len(recall_list) - 1):
                
                curr_recall = recall_list[recall_index]
                next_recall = recall_list[recall_index + 1]
                ip = interpolated_precision[recall_index]
            
                current_summation = (next_recall - curr_recall)*(ip)
                Avg_precision += current_summation
            
            AP.append(Avg_precision)
        
        #average all the AP
        mean_AvgPrecision = np.mean(np.asarray(AP, dtype=np.float32))
        
        return mean_AvgPrecision
            


    def calculate_iou(self, predicted_box, gt_boxes):
        '''
        Calculates the IoU between multiple ground truth boxes and a single predicted box.
        '''
        
        gt_boxes = np.asarray(gt_boxes, dtype=np.float32)
        
        x = np.minimum(gt_boxes[:,1] + gt_boxes[:,3], predicted_box[1] + predicted_box[3]) - np.maximum(gt_boxes[:,1], predicted_box[1])
        y = np.minimum(gt_boxes[:,2] + gt_boxes[:,4], predicted_box[2] + predicted_box[4]) - np.maximum(gt_boxes[:,2], predicted_box[2])
        
        intersection = np.maximum(x * y, 0) #if there is no overlap, the intersection will be either 0 or less. Hence we set it to 0.
        union = (gt_boxes[:, 3] * gt_boxes[:, 4]) + (predicted_box[3] * predicted_box[4])
        
        #returns a numpy array of shape [number of gt box] where each element represents the IoU between the predicted box and the ground-truth box.
        return intersection/union
             
        
    
    def get_box_coordinates(self, boxes, indexes=None):
        '''
        Based on the regression values, calculate back the [x,y,w,h] of the boxes. 
        '''
        
        batch_size = boxes.shape[0]
        batch_transformed_values = []
        
        if indexes is None: #when the ground-truth boxes are given
            
            for i in range(batch_size):
                
                gt_array = boxes[i]
                
                transformed_values = []
                
                occupied_array_indexes = np.nonzero(gt_array[:,:,:,0]) #get the indexes of the arrays that does not have 0 as their confidence value.
                
                num_of_occupied_arrays = occupied_array_indexes[0].shape[0] #num of objects.
                
                for j in range(num_of_occupied_arrays):
                    
                    #responsible grids and anchor index.
                    gridX = occupied_array_indexes[0][j]
                    gridY = occupied_array_indexes[1][j]
                    anchor_index = occupied_array_indexes[2][j]
                    
                    center_x = (gt_array[gridX][gridY][anchor_index][1]*self.subsampled_ratio) + (gridX*self.subsampled_ratio)
                    center_y = (gt_array[gridX][gridY][anchor_index][2]*self.subsampled_ratio) + (gridY*self.subsampled_ratio)
                    width = (self.anchors_list[gridX][gridY][anchor_index][3])*(math.e**(gt_array[gridX][gridY][anchor_index][3]))
                    height = (self.anchors_list[gridX][gridY][anchor_index][4])*(math.e**(gt_array[gridX][gridY][anchor_index][4]))

                    #Added an extra element at the end (init to 0) to keep track of the gt box that has been assigned to a predicted box for the 
                    #calculation of mAP.
                    #[prob, center_x,center_y,width,height, class index, mAP assignability]
                    transformed_values.append([gt_array[gridX][gridY][anchor_index][0], center_x, center_y, width, height, 
                                               gt_array[gridX][gridY][anchor_index][5], 0])
                
                batch_transformed_values.append(transformed_values)
        
        else: #when the topN predicted boxes are given
            
            for i in range(batch_size):
                
                predicted_array = boxes[i]
                
                transformed_values = []
                
                for j in range(self.top_N):
                    
                    #NOTE : indexes is a tuple where the first element: x-values, second element : y values, and third element: anchor index.
                    gridX = indexes[0][i][j]
                    gridY = indexes[1][i][j]
                    anchor_index = indexes[2][i][j]
                    
                    center_x = (predicted_array[j][1]*self.subsampled_ratio) + (gridX*self.subsampled_ratio)
                    center_y = (predicted_array[j][2]*self.subsampled_ratio) + (gridY*self.subsampled_ratio)
                    width = (self.anchors_list[gridX][gridY][anchor_index][3])*(math.e**(predicted_array[j][3]))
                    height = (self.anchors_list[gridX][gridY][anchor_index][4])*(math.e**(predicted_array[j][4]))
                    
                    transformed_values.append([predicted_array[j][0],center_x, center_y, width, height, np.argmax(predicted_array[j][5:])])
            
        
                batch_transformed_values.append(np.asarray(transformed_values, dtype=np.float32))
                
        return batch_transformed_values
                


    def get_topN_prediction(self, predicted_boxes):
        '''
        Returns the top N predicted array based on the confidence scores.
        NOTE: I could not find a way to sort ALL the boxes based on the confidence score using a numpy function. This is due to the fact that our 
                prediction array is in the shape of [batch size, feature map width, feature map height, number of anchors, 5 + number of classes] and if
                I use axis=4 (where the confidence is located.), it'll only sort the boxes in particular grids. It does not sort globally. Therefore, 
                I reshape the entire prediction array to [batch_size, number of boxes (or anchors)] and sort the confidence in decreasing order for each 
                batch. Finally, to get the indexes of the sorted boxes in terms of the original shape, i.e. [batch size, feature map width, feature map 
                height, number of anchors, 5 + number of classes], I used the information that there are k*feature_map_width boxes/anchors in each row 
                (feature map width : 0 to width-1) and there are k number of anchors/boxes in each grid.
        '''
        #get the batch size, feature size and the number of bounding boxes inside each grid.
        batch_size      = predicted_boxes.shape[0]
        feature_size    = self.feature_size
        num_of_anchor   = self.num_of_anchor
        
        #reshape the array with confidence scores only.
        flatten_pred = predicted_boxes[:,:,:,:,0].reshape(-1, feature_size*feature_size*num_of_anchor)
        
        #sort the arrays in ascending order and np.flip it to get it in the decreasing order.
        sorted_indexes = np.flip(flatten_pred.argsort(axis=1), axis=1)
        
        topN_indexes = sorted_indexes[:,:self.top_N] #Get the top N predictions from the entire batch.
        
        #use the defined function to convert the indexes to a grid-like indexes.
        cvtGenerator = lambda x:self.cvt_flattenedIndex2gridIndex(flattened_index=x)
        cvtFunction  = np.vectorize(cvtGenerator)
        
        #the converted indexes will be in the form of a tuple where the first element of the tuple shaped [batch_size, xgrids],
        #second element shaped [batch_size, ygrids] and the last element shaped [batch_size, anchor_indexes]
        converted_indexes = cvtFunction(topN_indexes)
        
        top_boxes = []
        
        #insert the indexes into each individual batch of boxes to obtain the top predicted boxes.
        for batch_index in range(batch_size):
            top_boxes.append(predicted_boxes[batch_index, converted_indexes[0][batch_index], converted_indexes[1][batch_index],
                                             converted_indexes[2][batch_index]])
        
        top_boxes = np.asarray(top_boxes)
        
        return top_boxes, converted_indexes



    def cvt_flattenedIndex2gridIndex(self, flattened_index):
        '''
        Converts the given index which was obtained from the flattened array back to the grid-like index.
        '''
        
        num_boxes_in_a_row = self.feature_size*self.num_of_anchor #number of boxes/anchors in an entire row of grids.
        
        xgrid, carryforward = divmod(flattened_index, num_boxes_in_a_row)
        ygrid, anchor_index = divmod(carryforward, self.num_of_anchor)
        
        return xgrid,ygrid,anchor_index
            
            
            


