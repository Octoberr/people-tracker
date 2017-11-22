# coding:utf-8
import numpy as np
import cv2
import os
from skimage import io

# face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# face_cascade.load('/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_default.xml')
# img = cv2.imread('faces.jpg')
img = io.imread('faces.jpg')
print img, type(img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# faces = face_cascade.detectMultiScale(gray, 1.3, 5)
#
# for (x,y,w,h) in faces:
#     cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
#
# cv2.imshow('img',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
