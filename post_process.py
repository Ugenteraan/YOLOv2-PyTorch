'''
This file contain modules that are useful for post-processing of YOLO such as calculation of mAP, non-max suppression, etc and modules related to them.
'''
import cfg
import numpy as np




class PostProcessing:
    '''
    Post Processing class contains modules to calculate mAP, non-max suppression, and modules related to these.
    '''
    
    def __init__(self, box_num_per_grid, feature_size, topN_pred, anchors_list):
        '''
        Initialize parameters. 
        box_num_per_grid : the number of anchors/boxes in a grid/cell.
        topN_pred : the top N prediction for mAP calculation.
        anchors_list : generated list of anchors used for the retrieval of the bounding box coordinates.
        '''
        
        self.num_of_anchor  = box_num_per_grid
        self.topN_pred      = topN_pred
        self.anchors_list   = anchors_list
        self.feature_size   = feature_size
        
        
    
    
    def calculate_meanAP(self, predicted_boxes, gt_boxes, anchors_list=self.anchors_list):
        '''
        1) Rank the predicted boxes in decreasing order based on the confidence score and choose top-N boxes.
        2) For every chosen boxes, determine if they are TP, FP or FN based on its IoU with the ground-truth boxes (There is no TN since an image 
        contains at least 1 object).
            a) If the chosen box has more than a certain set threshold with a ground-truth box and the class is also predicted correctly, it's a TP.
            b) If the chosen box has more than a certain set threshold with a ground-truth box but the class is predicted wrongly, it's a FN.
            c) If a ground-truth box is missed without a predicted box, then it's a FN.
            d) If the chosen box has less than a certain set threshold with a ground-truth box it's a FP regardless of the class prediction.
            e) If the chosen box has more than a certain set threshold with a ground-truth box, class is predicted correctly, but it's a duplicated
            prediction, then it's a FP.
        3) Calculate the recall and precision based on the TP, FP and FN for each class.
        4) Calculate the Interpolated Precision (Calculated at each recall level) for each class.
        5) Calculate Average Precision (AP) by taking the Area Under Curve of Interpolated Precision for each class.
        6) Mean the AP over all classes.
        '''

        top_boxes, top_indexes = self.get_topN_prediction(predicted_boxes)
        
        
        
    
    
    def get_box_coordinates(self, boxes, indexes=None, anchors_list=self.anchors_list):
        '''
        Based on the regression values, calculate back the [x,y,w,h] of the boxes. 
        '''
        
        
        
    
    
    def get_topN_prediction(self, predicted_boxes, top_N=self.topN_pred):
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
        shape           = predicted_boxes.shape
        batch_size      = shape[0]
        feature_size    = shape[1]
        num_of_anchor   = cfg.k
        
        #reshape the array with confidence scores only.
        flatten_pred = predicted_boxes[:,:,:,:,0].reshape(-1, feature_size*feature_size*num_of_anchor)
        
        #sort the arrays in ascending order and np.flip it to get it in the decreasing order.
        sorted_indexes = np.flip(flatten_pred.argsort(axis=1), axis=1)
        
        topN_indexes = sorted_indexes[:,:top_N] #Get the top N predictions from the entire batch.
        
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



    def cvt_flattenedIndex2gridIndex(self, flattened_index, feature_size=self.feature_size, num_of_anchor=self.num_of_anchor):
        '''
        Converts the given index which was obtained from the flattened array back to the grid-like index.
        '''
        
        num_boxes_in_a_row = feature_size*num_of_anchor #number of boxes/anchors in an entire row of grids.
        
        xgrid, carryforward = divmod(flattened_index, num_boxes_in_a_row)
        ygrid, anchor_index = divmod(carryforward, num_of_anchor)
        
        return xgrid,ygrid,anchor_index
            
            
            


