import numpy as np
import cv2
import json
from matplotlib import pyplot as plt
import os
from os import listdir



def read_this(image_file, gray_scale=False):
    image_src = cv2.imread(image_file)
    if gray_scale:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2GRAY)
    else:
        image_src = cv2.cvtColor(image_src, cv2.COLOR_BGR2RGB)
    return image_src


def binarize_lib(image_file, thresh_val=128, with_plot=False, gray_scale=False):
    image_src = read_this(image_file, gray_scale=gray_scale)
    th, image_b = cv2.threshold(src=image_src, thresh=thresh_val, maxval=255, type=cv2.THRESH_BINARY) 
    return image_b

def remove_noise_and_smooth(img):
    filtered = cv2.adaptiveThreshold(img.astype(np.uint8), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 41, 3)
    kernel = np.ones((1, 1), np.uint8)
    opening = cv2.morphologyEx(filtered, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    removed_img = cv2.bitwise_or(img, closing)

    return removed_img

def thinning_and_skeletonization(image):
    kernel = np.ones((5,5),np.uint8)
    skel = cv2.erode(image_noise,kernel,iterations = 1)
    return skel


folder_dir = "D:/researsh/images"

for images in os.listdir(folder_dir):
    if (images.endswith(".png") or images.endswith(".jpg")\
        or images.endswith(".jpeg")):
        downloadpath=r'D:\\researsh\\images\\'+ images
        savepath=r'D:\\researsh\\test\\'+ images
        img = cv2.imread(downloadpath)
        image_bin=binarize_lib(downloadpath, with_plot=True, gray_scale=True)
        image_noise=remove_noise_and_smooth(image_bin)
        image_thin=thinning_and_skeletonization(image_noise)
        cv2.imwrite(savepath,image_thin)






