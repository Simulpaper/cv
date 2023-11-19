import cv2
import numpy as np
import os

def process_image(img):
    bil = cv2.medianBlur(img, 3)
    # cv2.imshow('median blurred', bil) 
    # cv2.waitKey(0)

    # bil = cv2.fastNlMeansDenoising(bil, None, 30, 11, 41)
    # # cv2.imshow('denoise', bil) 
    # # cv2.waitKey(0)

    # bil = cv2.adaptiveThreshold(bil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
    #                                       cv2.THRESH_BINARY, 199, 5) 
    # # cv2.imshow('adapt thresh', bil) 
    # # cv2.waitKey(0)

    bil = cv2.fastNlMeansDenoising(bil, None, 30, 11, 11)
    cv2.imshow(f"bin image", bil)
    cv2.waitKey(0)
    bil = cv2.adaptiveThreshold(bil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 19, 5) 
    cv2.imshow(f"bin image", bil)
    cv2.waitKey(0)

    bil = cv2.medianBlur(bil, 3)

    edge = cv2.Canny(bil, 150, 300)
    cv2.imshow("canny edge", edge)
    cv2.waitKey(0)

   

dir_str = "../component_dataset"


for filename in os.listdir(dir_str):
    dataset_img = cv2.imread(f"{dir_str}/{filename}", cv2.IMREAD_GRAYSCALE)
    result = process_image(dataset_img)
