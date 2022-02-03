import cv2
import numpy as np
import matplotlib.pyplot as plt
from imutils import contours
from skimage import measure
import argparse
import imutils
import cmd

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True)
help = "path to the image file"
args = vars(ap.parse_args())
image = cv2.imread('irisler.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray, (11, 11), 0)
thresh = cv2.threshold(blurred, 255, 255, cv2.THRESH_BINARY)[1]
thresh = cv2.erode(thresh, None, iterations=2)
thresh = cv2.dilate(thresh, None, iterations=4)
labels = measure.label(thresh, background=0)
mask = np.zeros(thresh.shape, dtype="uint8")
for label in np.unique(labels):
    if label ==0:
        continue
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)
    if numPixels > 300:
        mask = cv2.add(mask, labelMask)
cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = contours.sort_contours(cnts)[0]
canny = cv2.Canny(blurred, 30, 150, 3)
dilated = cv2.dilate(canny, (1, 1), iterations=0)
for (i, c) in enumerate(cnts):
    (x, y, w, h) = cv2.boundingRect(c)
    ((cX, cY), radius) = cv2.minEnclosingCircle(c)
    cv2.circle(image, (int(cX), int(cY)), int(radius), (0, 0, 255), 3)
    cv2.putText(image, "#{}".format(i + 1), (x, y - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
(cnt, hierarchy) = cv2.findContours(dilated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
cv2.drawContours(rgb, cnt, -1, (0, 255, 0), 2)

cv2.imshow("Image", image)
cv2.waitKey(0)
Traceback (most recent call last):
line 32, in <module>
    cnts = contours.sort_contours(cnts)[0]
  File "C:\Users\ceyda\PycharmProjects\opencv-zone\venv\lib\site-packages\imutils\contours.py"
, line 24, in sort_contours
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
ValueError: not enough values to unpack (expected 2, got 0)

 
