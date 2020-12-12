# object detection
# source: https://www.pyimagesearch.com/2017/09/11/object-detection-with-deep-learning-and-opencv/


import argparse
import numpy as np
import cv2

ap = argparse.ArgumentParser()
ap.add_argument("-i","--image", required=True, help="path to image")
ap.add_argument("-p","--protxt", required=True, help="path to protofile")
ap.add_argument("-m","--model", required=True, help="path to model")
ap.add_argument("-c","--conf", required=True, type=float, default=0.2, help="probability threshold")
args = vars(ap.parse_args())

CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
"sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

net = cv2.dnn.readNetFromCaffe(args["protxt"], args["model"])

image = cv2.imread(args["image"])
(h,w)=image[:2]
blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 0.007843,(300, 300), 127.5)

net.setInput(blob)
detections = net.forward()
print(detections)
cv2.imshow(blob)