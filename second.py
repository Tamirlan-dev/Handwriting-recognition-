import cv2
import numpy as np
import os
from os import listdir

folder_dir = "D:/researsh/images"

for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        downloadpath=r'D:\\researsh\\images\\'+ images
        savepath=r'D:\\researsh\\test2\\'+ images
        img = cv2.imread(downloadpath)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (0,0), sigmaX=10, sigmaY=10)
        divide = cv2.divide(gray, blur, scale=255)
        thresh = cv2.threshold(divide, 240, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        #cv2.imwrite(r'D:\researsh\test\binarization2'+images,thresh)
        filtered = cv2.adaptiveThreshold(thresh.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
        kernel = np.ones((1, 1), np.uint8)
        opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
        removed_img = cv2.bitwise_or(thresh, closing)
        #cv2.imwrite(r'D:\researsh\test\remove_noise2'+images,removed_img)
        kernel = np.ones((2,2),np.uint8)
        skel = cv2.erode(removed_img,kernel,iterations = 1)
        cv2.imwrite(savepath, skel)






