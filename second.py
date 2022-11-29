import cv2
import numpy as np
import os
from os import listdir

folder_dir = "D:/researsh/images"

for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        downloadpath=r'D:\\researsh\\images\\'+ images
        print(downloadpath)
        savepath=r'D:\\researsh\\test\\'+ images
        img = cv2.imread(downloadpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0,0), sigmaX=33, sigmaY=33)
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        kernel = np.ones((5,5),np.uint8)
        skel = cv2.erode(morph,kernel,iterations = 5)
        cv2.imwrite(savepath, skel)






