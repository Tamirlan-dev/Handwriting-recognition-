import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
import os
from os import listdir


def img_bin_noise_thin(img):
    image=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    th, image_b = cv2.threshold(src=image, thresh=240, maxval=255, type=cv2.THRESH_BINARY) 
    filtered = cv2.adaptiveThreshold(image_b.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    removed_img = cv2.bitwise_or(image_b, closing)
    #cv2.imwrite(r'D:\researsh\test\remove_noise'+images,removed_img)
    kernel = np.ones((2,2),np.uint8)
    result = cv2.erode(removed_img,kernel,iterations = 1)
    #cv2.imwrite(r'D:\researsh\test\thinning'+images,result)
    return result


folder_dir = "D:/researsh/images"

for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        downloadpath=r'D:\\researsh\\images\\'+ images
        savepath=r'D:\\researsh\\test\\'+ images
        img = cv2.imread(downloadpath)
        result_img=img_bin_noise_thin(img)
        cv2.imwrite(savepath,result_img)




