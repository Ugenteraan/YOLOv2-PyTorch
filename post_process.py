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

import numpy as np
import torch
import cfg

class PostProcess:
    '''
    Contains modules to perform non-max suppressions and calculation of mAP.
    '''

    def __init__(self, box_num_per_grid, feature_size, topN_pred, anchors_list, iou_thresh=cfg.MAP_IOU_THRESH, confidence_thresh=cfg.CONFIDENCE_THRESH,
                 subsampled_ratio=cfg.SUBSAMPLED_RATIO, num_class=cfg.NUM_OF_CLASS, nms_iou_thresh=cfg.NMS_IOU_THRESH):
        '''
        Initialize parameters.
        box_num_per_grid : the number of anchors/boxes in a grid/cell.
        topN_pred : the top N prediction for mAP calculation.
        anchors_list : generated list of anchors used for the retrieval of the bounding box coordinates.
        '''

        self.num_of_anchor = box_num_per_grid
        self.top_n = topN_pred
        self.anchors_list = torch.from_numpy(anchors_list).to(cfg.DEVICE)
        self.feature_size = feature_size
        self.subsampled_ratio = subsampled_ratio
        self.iou_thresh = iou_thresh
        self.confidence_thresh = confidence_thresh
        self.nms_iou_thres = nms_iou_thresh
        self.epoch_predboxes = []
        self.epoch_gtboxes = []
        self.num_class = num_class

    def get_box_coordinates(self, network_prediction):
        '''
        Given the regression values from the prediction of the network, calculate back the predicted box's coordinate for the entire batch.
        '''

        #set the array values for the predicted probability lower than the threshold to 0
        boolean_array = network_prediction[:, :, :, :, 0] < self.confidence_thresh
        network_prediction[boolean_array] = 0

        #contains the grid indexes of the values in the pred array in the shape of [num_of_dimension(5), batch_size, xgrids, ygrids, anchor_index,25]
        #each dimension(axis 0) contains the index num of the following dimensions
        #[0] contains the batch index, [1] contains the xgrids, ... [4] contains the 25 length vectors.
        all_indices = torch.from_numpy(np.indices(dimensions=network_prediction.size())).to(cfg.DEVICE)

        all_x = network_prediction[:, :, :, :, 1:2] #all the x-coor regression values
        all_y = network_prediction[:, :, :, :, 2:3]
        all_w = network_prediction[:, :, :, :, 3:4]
        all_h = network_prediction[:, :, :, :, 4:5]

        anchors_w = self.anchors_list[:, :, :, 3:4]
        anchors_h = self.anchors_list[:, :, :, 4:5]

        grid_x = all_indices[1][:, :, :, :, 1:2]
        grid_y = all_indices[2][:, :, :, :, 2:3]

        cvt_w = anchors_w*(torch.exp(all_w))
        cvt_h = anchors_h*(torch.exp(all_h))
        cvt_x = (all_x*self.subsampled_ratio) + (grid_x*self.subsampled_ratio)
        cvt_y = (all_y*self.subsampled_ratio) + (grid_y*self.subsampled_ratio)

        #calculate the x1,y1,x2,y2 and insert them in the index 1,2,3 and 4 in the prediction array.
        network_prediction[:, :, :, :, 1:2] = cvt_x - cvt_w/2
        network_prediction[:, :, :, :, 2:3] = cvt_y - cvt_h/2
        network_prediction[:, :, :, :, 3:4] = cvt_x + cvt_w/2
        network_prediction[:, :, :, :, 4:5] = cvt_y + cvt_y/2

        #replace center x,y and w,h with x1,y1,x2,y2
        return network_prediction

    def nms(self, predictions):
        '''
        Perform Non-Max Suppression and returns the prediction arrays in suppressed form.
        '''

        batch_size = predictions.size()[0]
        cvt_arrays = self.get_box_coordinates(network_prediction=predictions)

        num_predictions = self.feature_size*self.feature_size*self.num_of_anchor

        #THE PREDICTION ARRAY IS TO BE RESHAPED TO [BATCH_SIZE, TOTAL_PREDICTION_BOXES, 5+NUM_CLASS]. ONCE THE RESHAPING IS DONE, THE ARRAY
        #WILL BE THEN SORTED ACCORDING TO THE CONFIDENCE VALUES WHICH IS LOCATED IN [:,:,0] THE FIRST INDEX IN THE THIRD AXIS.
        reshaped_pred = cvt_arrays.view(-1, num_predictions, 5+self.num_class)


        #convert the one-hot vector of the class label into the index of the class.
        reshaped_pred[:, :, 5] = torch.argmax(reshaped_pred[:, :, 5:], dim=-1)
        reshaped_pred = reshaped_pred[:, :, :6]



        sorted_index = torch.argsort(input=reshaped_pred[:, :, 0], descending=True) #sorting on the third axis, the location of the confidence scores.
        #We cannot simply insert the sorted_index variable into the reshaped_pred as this is a 3D array, not 2D. Simply inserting the indexes
        #will throw an error stating that the value is out of bound for axis 0 with size [BATCH_SIZE]. This is due to the fact that
        #the sorted_index contains the value of from 0 to num_predictions in the shape of [batch_size,num_predictions]. We cannot insert it with
        #reshaped_pred[:, sorted_index] either as it will create an extra dimension with the size of the [BATCH_SIZE].
        # Therefore, we need to specify the batch indices as well. Hence the torch.arange.
        sorted_pred = reshaped_pred[torch.arange(start=0, end=batch_size).view(-1, 1), sorted_index[:, :]]


        #zero-out the arrays that contains lower than the confidence threshold.
        boolean_array = sorted_pred[:, :, 0] < self.confidence_thresh
        sorted_pred[boolean_array] = 0

        #we will iterate through every prediction in all the batches and suppress the boxes that belongs to the same class of another box
        #that has a higher confidence and the IoU between them is more than the set threshold.
        for i in range(num_predictions-1): #-1 since the last prediction has nothing to be compared to.

            ref_pred = sorted_pred[:, i:i+1] # i+1 is to retain the dimensions.
            ref_class = sorted_pred[:, i:i+1, 5:6]

            comparing_arrays = sorted_pred[:, i+1:].contiguous() #make a copy of the arrays that we'll be comparing our reference pred to.

            #whichever prediction that does not belong to the same class as the reference pred array will be zeroed out.
            comparing_arrays = torch.where(comparing_arrays[:, :, 5:6] == ref_class, comparing_arrays, torch.Tensor([0.]).to(cfg.DEVICE))

            #get the iou between the reference pred array and all other remaining arrays on the right.
            iou_batch = iou_check(box_a=ref_pred, box_b=comparing_arrays)

            #NOTE that we're not using comparing_arrays in the torch.where because the arrays that do not have the same class
            #with the reference array were zeroed. Whichever array that has the iou more than the threshold, the confidence value
            #will be zeroed.
            sorted_pred[:, i+1:, 0] = torch.where((iou_batch < self.nms_iou_thres) , sorted_pred[:, i+1:, 0], torch.Tensor([0.]).to(cfg.DEVICE))

        #zero the entire array that has confidence lower than the threshold again.
        boolean_array = sorted_pred[:, :, 0] < self.confidence_thresh
        sorted_pred[boolean_array] = 0

        return sorted_pred




def iou_check(box_a, box_b):
    '''
    Calculate the IoU between a batch of single array with the same batch of the remaining arrays on the right.
    '''

    x_a = torch.max(box_a[:, :, 1], box_b[:, :, 1])
    y_a = torch.max(box_a[:, :, 2], box_b[:, :, 2])
    x_b = torch.min(box_a[:, :, 3], box_b[:, :, 3])
    y_b = torch.min(box_a[:, :, 4], box_b[:, :, 4])

    ref_tensor = torch.Tensor([0.]).to(cfg.DEVICE)
    inter_area_noadd = torch.max(ref_tensor, x_b-x_a) * torch.max(ref_tensor, y_b-y_a)

    #Since adding one to make up for the 0-indexing would cause boxes with 0 coordinates to have 1 as interArea, we'll implement
    #torch.where to add 1 only when the the value of the element is not 0.
    inter_area_added = torch.max(ref_tensor, x_b-x_a+1) * torch.max(ref_tensor, y_b-y_a+1)
    #torch where keeps the elements when it's true and replace with the second array given when it's false.
    inter_area = torch.where(inter_area_noadd == 0, inter_area_noadd, inter_area_added)

    #we can add 1 safely here since an intersection area of 0 would yield 0 when divided anyways.
    box_a_area = (box_a[:, :, 3] - box_a[:, :, 1]+1) * (box_a[:, :, 4] - box_a[:, :, 2]+1)
    box_b_area = (box_b[:, :, 3] - box_b[:, :, 1]+1) * (box_b[:, :, 4] - box_b[:, :, 2]+1)

    iou = inter_area / (box_a_area + box_b_area - inter_area)

    return iou