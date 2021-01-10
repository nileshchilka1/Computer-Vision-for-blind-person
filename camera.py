import cv2
from collections import Counter
import time
import numpy as np
import base64
from model.yolo_model import YOLO
import gc

yolo = None
all_classes = None



language='en' #THE LANGUAGE FOR THE SPEECH DATA
def process_image(image):
    """Resize, reduce and expand image.

    # Argument:
        img: original image.

    # Returns
        image: ndarray(64, 64, 3), processed image.
    """
    
    image = cv2.resize(image, (416, 416),
                       interpolation=cv2.INTER_CUBIC)
    image = np.array(image, dtype='float32')
    image /= 255.
    image = np.expand_dims(image, axis=0)
    print('image_processed')
    return image

def get_classes(file):
    """Get classes name.

    # Argument:
        file: classes name for database.

    # Returns
        class_names: List, classes name.

    """
    with open(file) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]

    return class_names


def draw(image, boxes, scores, classes, all_classes):
    """Draw the boxes on the image.

    # Argument:
        image: original image.
        boxes: ndarray, boxes of objects.
        classes: ndarray, classes of objects.
        scores: ndarray, scores of objects.
        all_classes: all classes name.
    """
    locations=[[],[],[],[],[],[],[],[],[]]
    for box, score, cl in zip(boxes, scores, classes):
        x, y, w, h = box

        top = max(0, np.floor(x + 0.5).astype(int))
        left = max(0, np.floor(y + 0.5).astype(int))
        right = min(image.shape[1], np.floor(x + w + 0.5).astype(int))
        bottom = min(image.shape[0], np.floor(y + h + 0.5).astype(int))
        center_x = int(top+((right-top)/2))
        center_y = int(left+((bottom-left)/2))
        cv2.rectangle(image, (top, left), (right, bottom), (255, 0, 0), 2)
        cv2.circle(image,(center_x,center_y) , 2, (0, 255, 0), 2)
        
        if center_x < 170:
            if center_y <130:
                locations[0].append(all_classes[cl])
            elif center_y < 350:
                locations[3].append(all_classes[cl])
            else: 
                locations[6].append(all_classes[cl])
        elif center_x < 430:
            if center_y <130:
                locations[1].append(all_classes[cl])
            elif center_y < 350:
                locations[4].append(all_classes[cl])
            else: 
                locations[7].append(all_classes[cl])
        else:
            if center_y <130:
                locations[2].append(all_classes[cl])
            elif center_y < 350:
                locations[5].append(all_classes[cl])
            else: 
                locations[8].append(all_classes[cl])
                
        
        cv2.putText(image, '{0} {1:.2f}'.format(all_classes[cl], score),
                    (top, left - 6),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, (0, 0, 255), 1,
                    cv2.LINE_AA)

        print('class: {0}, score: {1:.2f}'.format(all_classes[cl], score))
        print('box coordinate x,y,w,h: {0}'.format(box))

    print()
    return locations

def detect_image(image):
    """Use yolo v3 to detect images.

    # Argument:
        image: original image.
        yolo: YOLO, yolo model.
        all_classes: all classes name.

    # Returns:
        image: processed image.
    """
    global yolo,all_classes
    
    if yolo is None or all_classes is None:
        load_saved_artifacts()
    
    k=[]
    pimage = process_image(image)
    
    start = time.time()
    print('predicting')
    
    boxes, classes, scores = yolo.predict(pimage, image.shape)
    end = time.time()

    print('time: {0:.2f}s'.format(end - start))

    if boxes is not None:

        locations=draw(image, boxes, scores, classes, all_classes)
        
    del pimage
    gc.collect()
    print(locations)
    return image,locations


def get_cv2_image_from_base64_string(b64str):
    '''
    credit: https://stackoverflow.com/questions/33754935/read-a-base-64-encoded-image-from-memory-using-opencv-python-library
    :param uri:
    :return:
    '''
    b64str = b64str.split(',')[1]
    b64str = np.frombuffer(base64.b64decode(b64str), np.uint8)
    b64str = cv2.imdecode(b64str, cv2.IMREAD_COLOR)
    return b64str

def load_saved_artifacts():
    print("loading saved artifacts...start")

    global yolo
    global all_classes
    if yolo is None or all_classes is None:
        
        yolo = YOLO(0.6, 0.5)
        all_classes = get_classes('data/coco_classes.txt')
            
    print("loading saved artifacts...done")
        

def vision(img):
    
        img = get_cv2_image_from_base64_string(img)
        
        img,locations=detect_image(img)
        
        location_text =  ['The Objects at top left are,','The Objects at top are,','The Objects at top right are,','The Objects at center left are,','The Objects at center are,','The Objects at center right are,','The Objects at bottom left are,','The Objects at bottom are,','The Objects at bottom right are,']
        
        mytext = ''
        for i,k in enumerate(locations):
            count_of_object = Counter(k)
        
            object_count = ''
            if len(count_of_object) != 0:
                    for key,value in count_of_object.items():
                        
                        object_count += str(value) +' '+ str(key)+ ' '
                        
                        if mytext == '':
                            mytext = location_text[i] + object_count
                        else:
                            
                            mytext +=  location_text[i] + object_count
                            
            
        
       
        print(mytext)
        
        cv2.line(img,(170,0),(170,480),(0,0,255),1)
        cv2.line(img,(430,0),(430,480),(0,0,255),1)
        cv2.line(img,(0,130),(600,130),(0,0,255),1)
        cv2.line(img,(0,350),(600,350),(0,0,255),1)
        
        #cv2.imshow('cam',img)
        #cv2.waitKey()
               
        #cv2.destroyAllWindows()
        ret, img = cv2.imencode('.jpg', img)

        img = img.tobytes()
        
        img = base64.b64encode(img)
        del object_count,count_of_object,ret,k
        gc.collect()
        return img,mytext


if __name__ == '__main__':
    load_saved_artifacts()
    
    

















