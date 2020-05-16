import xmltodict
import numpy as np
from K_Means import K_Means
import cv2
from label_format import label_formatting
import glob
import os.path




def read_image(image_path, resized_image_size):
    '''
    Read a single image from the given index into a numpy array.
    '''

    im_ = cv2.imread(image_path)
    im_ = cv2.cvtColor(im_, cv2.COLOR_BGR2RGB)
    im_ = cv2.resize(im_, (resized_image_size, resized_image_size))
    img = im_/255 #normalize image

    return np.asarray(img, dtype=np.float32)


def ImgNet_check_model(model_path):
    '''
    Checks if the ImageNet trained model is present or not.
    '''
    
    if os.path.isfile(model_path):
        return True 

    return False


def ImgNet_process_classname(classname):
    '''
    Process the given string.
    '''
    
    processed_str = classname.replace("'","")
    processed_str = processed_str.replace(" ", "_")
    processed_str = processed_str.lower()
    
    return processed_str
    

def ImgNet_get_classes(folder_path):
    '''
    Gets all the folder names (classes) of the ImageNet dataset.
    '''
    class_list = []
    for path in glob.glob(folder_path + "/**"):
        
        class_name = path.split('/')[-1] #the class name would be at the back of the last slash.
        class_name = ImgNet_process_classname(classname = class_name)
        class_list.append(class_name)
    
    return sorted(class_list)


def ImgNet_generate_data(folder_path, class_list):
    '''
    Build a list of image data and corresponding label.
    '''
    
    images_list = []
    labels_list = []
    
    for path in glob.glob(folder_path + "/**", recursive=True):
        
        if path[-3:] == 'jpg':
            images_list.append(path)
        
            classname = path.split('/')[-2]
            classname = ImgNet_process_classname(classname=classname)
            class_index = class_list.index(classname)
            
            labels_list.append(class_index)
    
    return images_list, labels_list


def ImgNet_read_data(image_path, class_idx, resized_image_size):
    '''
    Reads a single image and the corresponding label and wraps them in a numpy array.
    '''
    
    image_array = read_image(image_path=image_path, resized_image_size=resized_image_size)
    label_array = np.asarray(class_idx, dtype=np.int32)
    
    return image_array, label_array




def get_classes(xml_files):
    '''
    Gets all the classes from the dataset.
    '''
    classes = []

    for file in xml_files:

        f = open(file)
        doc = xmltodict.parse(f.read()) #parse the xml file contents to python dict.
        #Images in the dataset might contain either 1 object or more than 1 object. For images with 1 object, the annotation for the object
        #in the xml file will be located in 'annotation' -> 'object' -> 'name'. For images with more than 1 object, the annotations for the objects
        #will be nested in 'annotation' -> 'object' thus requiring a loop to iterate through them. (Pascal VOC format)

        try: 
            #try iterating through the tag. (For images with more than 1 obj.)
            for obj in doc['annotation']['object']:
                classes.append(obj['name'].lower()) #append the lowercased string.

        except TypeError: #iterating through non-nested tags would throw a TypeError.
            classes.append(doc['annotation']['object']['name'].lower()) #append the lowercased string.

        f.close()

    classes = list(set(classes)) #remove duplicates.
    classes.sort() #to maintain consistency.

    #returns a list containing the names of classes after being sorted.
    return classes




def get_labels_from_xml(xml_file_path, classes, resized_image_size, excluded_classes):
    '''
    Input : A SINGLE xml file and the total number of classes in the dataset. 
    Output: Labels in numpy array format (Object classes their corresponding bounding box coordinates).

    Desc : This function parses a single xml file and outputs the objects classes and their corresponding bounding box coordinates
        [top-left-x, top-left-y, btm-right-x, btm-right-y] on the resized image.

    '''

    f = open(xml_file_path)
    doc = xmltodict.parse(f.read()) #parse the xml file to python dict.

    #get the original image height and width. Images have different height and width from each other.
    ori_img_height = float(doc['annotation']['size']['height'])
    ori_img_width  = float(doc['annotation']['size']['width'])


    class_label = [] #init for keeping track objects' labels.
    bbox_label  = [] #init for keeping track of objects' bounding box (bb).


    #Images in the dataset might contain either 1 object or more than 1 object. For images with 1 object, the annotation for the object
    #in the xml file will be located in 'annotation' -> 'object' -> 'name'. For images with more than 1 object, the annotations for the objects
    #will be nested in 'annotation' -> 'object' thus requiring a loop to iterate through them. (Pascal VOC format)
    try:
        #Try iterating through the tag (For images with more than 1 obj).
        for each_obj in doc['annotation']['object']:
            
            obj_class = each_obj['name'].lower() #get the label for the object and lowercase the string.
            
            if obj_class in excluded_classes:
                continue

            #Pascal VOC's format to denote bounding boxes are to denote the top left part of the box and the bottom right of the box.
            #the coordinates are in terms of x and y axis for both part of the box.
            x_min = float(each_obj['bndbox']['xmin']) #top left x-axis coordinate.
            x_max = float(each_obj['bndbox']['xmax']) #bottom right x-axis coordinate.
            y_min = float(each_obj['bndbox']['ymin']) #top left y-axis coordinate.
            y_max = float(each_obj['bndbox']['ymax']) #bottom right y-axis coordinate.

        ##################################################################################
        #We want to make sure the coordinates are resized according to the resized image.#
        ##################################################################################

            #All the images will be resized to a fixed size in order to be fixed-size inputs to the neural network model.
            #Therefore, we need to resize the coordinates as well since the coordinates above is based on the original size of the images.

            #In order to find the resized coordinates, we must multiply the ratio of the resized image compared to its original to the coordinates.
            x_min = float((resized_image_size/ori_img_width)*x_min)
            y_min = float((resized_image_size/ori_img_height)*y_min)
            x_max = float((resized_image_size/ori_img_width)*x_max)
            y_max = float((resized_image_size/ori_img_height)*y_max)

            generated_box_info = [x_min, y_min, x_max, y_max]


            index = classes.index(obj_class) #get the index of the object's class.

            #append each object's class label and the bounding box label (converted to Faster R-CNN format) into the list initialized earlier.
            class_label.append(index)
            bbox_label.append(np.asarray(generated_box_info, dtype='float32'))

    except TypeError : #happens when the iteration through the tag fails due to only 1 object being in the image.
        
        #SAME PROCEDURE AS ABOVE !  

        #Getting these information from the XML file differs compared to above,
        obj_class = doc['annotation']['object']['name']
        
        if not obj_class in excluded_classes:
                        
            x_min = float(doc['annotation']['object']['bndbox']['xmin']) 
            x_max = float(doc['annotation']['object']['bndbox']['xmax']) 
            y_min = float(doc['annotation']['object']['bndbox']['ymin']) 
            y_max = float(doc['annotation']['object']['bndbox']['ymax']) 

            x_min = float((resized_image_size/ori_img_width)*x_min)
            y_min = float((resized_image_size/ori_img_height)*y_min)
            x_max = float((resized_image_size/ori_img_width)*x_max)
            y_max = float((resized_image_size/ori_img_height)*y_max)

            generated_box_info = [x_min, y_min, x_max, y_max]

            #Get the index of the class
            index = classes.index(obj_class) 

            class_label.append(index)
            bbox_label.append(np.asarray(generated_box_info, dtype='float32'))


    return class_label, np.asarray(bbox_label)



def cluster_bounding_boxes( k, total_images, resized_image_size, list_annotations, classes, excluded_classes):
    ''' 
    Use modified K-Means algorithm to find the best k anchor box sizes.
    '''

    gt_boxes_array = []
    
    for i in range(total_images): 

        #extract the class label (unecessary) and ground-truth boxes for the specific image.
        _, gt_boxes = get_labels_from_xml(xml_file_path=list_annotations[i], classes=classes,
                                                        resized_image_size=resized_image_size, excluded_classes=excluded_classes)
        
        #in order to treat each bounding box as one data point, we have to extract every bounding box from the label files.
        #some images only contain 1 object whereas other images contain more than 1 objects.
        if len(gt_boxes) == 1:
            gt_boxes_array.append(np.asarray(gt_boxes[0], dtype=np.float32))
        else:
            for j in range(len(gt_boxes)):
                gt_boxes_array.append(np.asarray(gt_boxes[j], dtype=np.float32))
    
    #convert the list into numpy arrays
    gt_boxes_array = np.asarray(gt_boxes_array, dtype=np.float32)
    
    kmeans = K_Means(k=k, boxes=gt_boxes_array)
    anchor_sizes = kmeans()
    anchor_sizes = np.asarray(anchor_sizes, dtype=np.int32) #convert to integer
    
    #k anchors
    return anchor_sizes


def generate_anchors(anchor_sizes, detection_conv_size, subsampled_ratio, resized_image_size):
    '''
    Place anchors consistently on every grid of the subsampled image. The anchors are however, referenced to the original resized image.
    '''
    

    subsampled_image_size = int(resized_image_size/subsampled_ratio)
    
    #each grid has len(anchor_sizes) anchors and each anchor has 5 elements.
    #the first element denotes the x-grid and the second element denotes the y-grid.
    #the third element denotes the i-th anchor and the last element denotes the elements of the i-th anchor.
    anchors_list  = np.zeros((subsampled_image_size, subsampled_image_size, len(anchor_sizes), 5), dtype=np.float32)

    anchor_center = [0, 0]
    
    #iteration stops when the index goes 1 step beyond the size of the feature map.
    while (anchor_center != [0, subsampled_image_size]):
        
        #access each anchor size 
        for index, anchor in enumerate(anchor_sizes):

            #the anchors are referenced to the original image.
            anchor_coor = [anchor_center[0]*subsampled_ratio, anchor_center[1]*subsampled_ratio,
                        anchor[0], anchor[1]]
            
            anchors_list[anchor_center[0], anchor_center[1], index, :] = [0] + anchor_coor
        
        anchor_center[0] += 1
        
        #if the width of the image has exceeded.
        if anchor_center[0] == subsampled_image_size :
            
            anchor_center[1] += 1
            anchor_center[0] = 0
        
    
    return anchors_list




def generate_training_data(data_index, anchors_list, xml_file_path, classes, resized_image_size, subsampled_ratio, excluded_classes, image_path):
    '''
    Returns the image array and the corresponding label in the required format based for the given index.
    '''

    #get the label(s) and ground-truth bounding box(es) (x1,y1,x2,y2) for a given xml file path.
    object_labels, gt_boxes = get_labels_from_xml(xml_file_path=xml_file_path, resized_image_size=resized_image_size,
                                                                        classes=classes, excluded_classes=excluded_classes)
    

    image_array = read_image(image_path=image_path, resized_image_size=resized_image_size)

    #label formatting
    label_array = label_formatting(gt_class_labels=object_labels, gt_boxes=gt_boxes, anchors_list=anchors_list,
                                                            subsampled_ratio=subsampled_ratio, resized_image_size=resized_image_size, classes=classes)

    
    
    return (image_array, label_array)    