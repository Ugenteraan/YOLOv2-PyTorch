'''
Non-Max Suppression and mAP calculation.

1) NMS Process:

Input : Prediction tensor, pred shaped [batch_size, xgrids(feature_size), ygrids(feature_size), anchor_nums_in_a_box(k), 5+class_num]
Output: Tensor shaped [batch_size, total_anchor_nums, 5+class_num] where the suppressed arrays will have values of 0 and the remaining
        arrays will be converted to the box coordinates instead of regression values as the input arrays.

    i)    If pred[:,:,:,:,0] < confidence_threshold : #if the confidence score of an array is lower than the threshold
            pred[:,:,:,i] = 0                         #zero-out the entire array.
    ii)   Convert ENTIRE pred regression values into box coordinates. 
    iii)  Reshape pred -> [batch_size, total_anchor_nums(xgrids*ygrids*anchor_nums_in_a_box), 5+class_num]
    iv)  Sort pred based on the confidence scores in the descending order.
    v)   FOR j from 0 to total_anchor_nums DO
             ref_box = pred[:, j]
             check IoU between ref_box and all other prediction arrays, pred[:, j+1:]
             IF the IoU between the ref box and the compared_box is > IoU_Threshold AND class of ref_box & class of compared_box are SAME :
                compared_box = 0             
'''
import cfg
import numpy as np
import torch


class PostProcess:
        
    def __init__(self, box_num_per_grid, feature_size, topN_pred, anchors_list, IoU_thresh=cfg.mAP_iou_thresh, confidence_thresh=cfg.confidence_thresh, 
                                                    subsampled_ratio=cfg.subsampled_ratio, num_class=cfg.num_of_class, nms_iou_thresh=cfg.nms_iou_thresh):
        '''
        Initialize parameters. 
        box_num_per_grid : the number of anchors/boxes in a grid/cell.
        topN_pred : the top N prediction for mAP calculation.
        anchors_list : generated list of anchors used for the retrieval of the bounding box coordinates.
        '''
        
        self.num_of_anchor      = box_num_per_grid
        self.top_N              = topN_pred
        self.anchors_list       = torch.from_numpy(anchors_list).to(cfg.device)
        self.feature_size       = feature_size
        self.subsampled_ratio   = subsampled_ratio
        self.iou_thresh         = IoU_thresh
        self.confidence_thresh  = confidence_thresh
        self.nms_iou_thres      = nms_iou_thresh
        self.epoch_predboxes    = []
        self.epoch_gtBoxes      = []
        self.num_class          = num_class
        
    def get_box_coordinates(self, network_prediction):
        '''
        Given the regression values from the prediction of the network, calculate back the predicted box's coordinate for the entire batch.
        '''
        
        tensor_size = network_prediction.size() #number of data in the batch
        
        #set the array values for the predicted probability lower than the threshold to 0
        boolean_array = network_prediction[:,:,:,:,0] < self.confidence_thresh
        network_prediction[boolean_array] = 0 
        
        #contains the grid indexes of the values in the pred array in the shape of [num_of_dimension(5), batch_size, xgrids, ygrids, anchor_index,25]
        #each dimension(axis 0) contains the index num of the following dimensions
        #[0] contains the batch index, [1] contains the xgrids, ... [4] contains the 25 length vectors.
        all_indices = torch.from_numpy(np.indices(dimensions=tensor_size)).to(cfg.device)
        
        all_x = network_prediction[:,:,:,:,1:2] #all the x-coor regression values
        all_y = network_prediction[:,:,:,:,2:3]
        all_w = network_prediction[:,:,:,:,3:4]
        all_h = network_prediction[:,:,:,:,4:5]
        
        anchors_w = self.anchors_list[:,:,:,3:4]
        anchors_h = self.anchors_list[:,:,:,4:5]
        
        gridX = all_indices[1][:,:,:,:,1:2]
        gridY = all_indices[2][:,:,:,:,2:3]
        
        cvt_w = anchors_w*(torch.exp(all_w))
        cvt_h = anchors_h*(torch.exp(all_h))
        cvt_x = (all_x*self.subsampled_ratio) + (gridX*self.subsampled_ratio) 
        cvt_y = (all_y*self.subsampled_ratio) + (gridY*self.subsampled_ratio) 
        
        
        x1 = cvt_x - cvt_w/2
        y1 = cvt_y - cvt_h/2
        x2 = cvt_x + cvt_w/2
        y2 = cvt_y + cvt_y/2
        
        network_prediction[:,:,:,:,1:2] = x1
        network_prediction[:,:,:,:,2:3] = y1
        network_prediction[:,:,:,:,3:4] = x2
        network_prediction[:,:,:,:,4:5] = y2
        
        #replace center x,y and w,h with x1,y1,x2,y2
        return network_prediction


    def iou_check(self, boxA, boxB):
        '''
        Calculate the IoU between a batch of single array with the same batch of many arrays. If the confidence is 0, the IoU should be 0 as well.
        '''
        
        xA = torch.max(boxA[:,:,1], boxB[:,:,1])
        yA = torch.max(boxA[:,:,1], boxB[:,:,1])
        xB = torch.min(boxA[:,:,2], boxB[:,:,2])
        yB = torch.min(boxA[:,:,3], boxB[:,:,3])
        
        ref_tensor = torch.Tensor([0.]).to(cfg.device)
        interArea = torch.max(ref_tensor, xB-xA+1) * torch.max(ref_tensor, yB-yA+1)
        
        
        print(interArea.size())
        print(boxB.size())
        
        return None
        
            

    def nms(self, predictions):
        
        batch_size = predictions.size()[0]
        cvt_arrays = self.get_box_coordinates(network_prediction=predictions)
        
        num_predictions = self.feature_size*self.feature_size*self.num_of_anchor
        
        #THE PREDICTION ARRAY IS TO BE RESHAPED TO [BATCH_SIZE, TOTAL_PREDICTION_BOXES, 5+NUM_CLASS]. ONCE THE RESHAPING IS DONE, THE ARRAY
        #WILL BE THEN SORTED ACCORDING TO THE CONFIDENCE VALUES WHICH IS LOCATED IN [:,:,0] THE FIRST INDEX IN THE THIRD AXIS.
        reshaped_pred = cvt_arrays.view(-1,num_predictions, 5+self.num_class)
        
        
        #convert the one-hot vector of the class label into the index of the class.
        reshaped_pred[:,:,5] = torch.argmax(reshaped_pred[:,:,5:], dim=-1)
        reshaped_pred = reshaped_pred[:,:,:6]
        
        
        
        sorted_index = torch.argsort(input=reshaped_pred[:,:,0], descending=True) #sorting on the third axis, the location of the confidence scores.
        #We cannot simply insert the sorted_index variable into the reshaped_pred as this is a 3D array, not 2D. Simply inserting the indexes
        #will throw an error stating that the value is out of bound for axis 0 with size [BATCH_SIZE]. This is due to the fact that
        #the sorted_index contains the value of from 0 to num_predictions in the shape of [batch_size,num_predictions]. We cannot insert it with 
        #reshaped_pred[:, sorted_index] either as it will create an extra dimension with the size of the [BATCH_SIZE]. 
        # Therefore, we need to specify the batch indices as well. Hence the torch.arange.
        sorted_pred = reshaped_pred[torch.arange(start=0,end=batch_size).view(-1,1), sorted_index[:,:]]
        
        for i in range(num_predictions-1): #-1 since the last prediction has nothing to be compared to.
            
            ref_pred = sorted_pred[:,i:i+1] # i+1 is to retain the dimensions.
            
            iou_batch = self.iou_check(boxA=ref_pred, boxB=sorted_pred[:,i+1:])
        
        
        
        