'''
Created on 07.09.2020

@author: Sarah Boening
Basierend auf: https://github.com/opencv/opencv/blob/8c25a8eb7b10fb50cda323ee6bec68aa1a9ce43c/samples/dnn/object_detection.py
Tutorial fuer Objekterkennung mit OpenCV und Yolo: https://www.learnopencv.com/deep-learning-based-object-detection-using-yolov3-with-opencv-python-c/
Yolo-Netzwerk: https://github.com/AlexeyAB/darknet/wiki/YOLOv4-model-zoo
'''
import numpy as np
import cv2

def get_output_layers(net):  
    '''
    Definiert die Output-Layer, die net durchlaufen soll, hier alle
    '''
    layer_names = net.getLayerNames() # Namen aller Layer im Netzwerk
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()] # unconnected Layers = Output-Layers
    return output_layers # alle Output-Layer

def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h, classes):
    '''
    Zeichnet Bounding Box um erkannte Objekte und schreibt Klassenname dazu
    '''
    # zufaellige Farben fuer jede Klasse
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3)) # uniform = Gleichverteilung

    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
if __name__ == '__main__':
    '''
    1. Daten definieren
    '''
    network_weights = ".\\data\\yolov4.weights"
    network_config = ".\\data\\yolov4.cfg"
    image = ".\\data\\Josie.jpg"
    labels = ".\\data\\imagenet_1000_labels.txt"
    labels = ".\\data\\yolo_classes.txt"
    
    '''
    2. Bild laden
    '''
    image = cv2.imread(image)
    # optional Bild verkleinern
    scale_percent = 45
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    image = cv2.resize(image, dim) 
    # Bild anzeigen
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    '''
    3. Klassifizierungsklassen laden
    '''
    classes = open(labels).read().split("\n")
    print(classes[:10])
          
    '''
    3. Fertiges Netzwerk laden
    '''
    net = cv2.dnn.readNet(network_weights, network_config)

    # Netzwerk braucht spezielle Bildgroesse
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    '''
    4. Netzwerk laufen lassen
    '''
    net.setInput(blob)
    preds = net.forward(get_output_layers(net))
    
    '''
    5. Output post-processing
    '''
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # Fuer jedes erkannt Objekt von jedem Output-Layer brauchen wir:
    # die Confidence, Klassen-ID, und Bounding box
    # Objekte mit Confidence < 0.5 werden verworfen
    for out in preds:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # Non-Maxima Suppression - Filtert Output (fuer mehr Infos: https://towardsdatascience.com/non-maximum-suppression-nms-93ce178e177c)
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    # Geht alle gefilterten Vorschlaege durch
    for i in indices:
        i = i[0] # Klassenindex
        label = classes[class_ids[i]] # Name zu Index 
        box = boxes[i] # Box um Objekt
        x = round(box[0])
        y = round(box[1])
        w = round(box[2]) # width
        h = round(box[3]) # height
        print("Recognized object: ", label) # identifiziertes Objekt
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes) # Box um Objekt
    # Fertiges Bild anzeigen
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    
