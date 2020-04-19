import xmltodict



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
