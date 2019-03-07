# import the necessary packages
import numpy as np
import argparse
import time
import cv2
import os


CONFIDENCE = 0.5
THRESHOLD = 0.3

labelsPath = 'yolo-coco/coco.names'
LABELS = open(labelsPath).read().strip().split("\n")

weightsPath ='yolo-coco/yolov3.weights'
configPath ='yolo-coco/yolov3.cfg'

print("[INFO] loading YOLO from disk...")
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)

path ='images/'
images = sorted(os.listdir(path))
for image in images:
    img = cv2.imread(path+ image)
#     img = cv2.imread('images/4.jpg')
    (H, W) = img.shape[:2]
    # determine only the *output* layer names that we need from YOLO
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),swapRB=True, crop=False)
    net.setInput(blob)
    start = time.time()
    layerOutputs = net.forward(ln)
    end = time.time()

    # show timing information on YOLO
    # print("[INFO] YOLO took {:.6f} seconds".format(end - start))

    # initialize a list of colors to represent each possible class label
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype="uint8")

    # initialize our lists of detected bounding boxes, confidences, and
    # class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []
    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract the class ID and confidence (i.e., probability) of
            # the current object detection
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            if confidence > CONFIDENCE:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # update our list of bounding box coordinates, confidences,
                # and class IDs
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)


    # apply non-maxima suppression to suppress weak, overlapping bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONFIDENCE, THRESHOLD)
    # print('tagged images are :')
    if len(idxs) > 0:
        label=set()
    # loop over the indexes we are keeping
        for i in idxs.flatten():
            # extract the bounding box coordinates
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            # draw a bounding box rectangle and label on the image
            color = [int(c) for c in COLORS[classIDs[i]]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            text = "{}".format(LABELS[classIDs[i]])
            x= int((x+(x+w))/2)
            y= int((y+(y+h))/2)
            cv2.putText(img, text, ( x,y ), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            label.add(text)
#         print(label)
        stri = ', '.join(str(e) for e in label)
        print(stri)
        cv2.imshow("Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()