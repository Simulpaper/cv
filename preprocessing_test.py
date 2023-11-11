import cv2
import numpy as np
import os

def process_image(img):
    bil = cv2.medianBlur(img, 3)
    cv2.imshow('median blurred', bil) 
    cv2.waitKey(0)

    bil = cv2.fastNlMeansDenoising(bil, None, 30, 11, 21)
    cv2.imshow('denoise', bil) 
    cv2.waitKey(0)

    bil = cv2.adaptiveThreshold(bil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 191, 5) 
    cv2.imshow('adapt thresh', bil) 
    cv2.waitKey(0)

    bil = cv2.medianBlur(bil, 5)
    cv2.imshow('med blurr again', bil) 
    cv2.waitKey(0)

    # bil = cv2.GaussianBlur(bil,(5,5),0)
    # cv2.imshow('gaussian', bil) 
    # cv2.waitKey(0)

    # bil = cv2.fastNlMeansDenoising(bil, None, 30, 11, 21)
    # cv2.imshow('denoise', bil) 
    # cv2.waitKey(0)

    # bil = cv2.medianBlur(bil, 3)
    # cv2.imshow('med blurr again', bil) 
    # cv2.waitKey(0)

    edge = cv2.Canny(bil, 150, 300)
    cv2.imshow("canny edge", edge)
    cv2.waitKey(0)

    rows = img.shape[0]
    cols = img.shape[1]
    circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT, 1, max(rows, cols),
    param1=300, param2=30,
    minRadius=(min(rows, cols) // 5), maxRadius=(min(rows, cols) // 2))

    if circles is not None:
        circles = circles.astype(int)
        circles = circles[0]
    else:
        print("NO CIRCLES DETECTED!!")
        cv2.destroyAllWindows() 
        return

    circles_img = bil
    for i in range(len(circles)):
        circles_img = cv2.circle(circles_img, (circles[i][0], circles[i][1]), circles[i][2], (0, 100, 100), 3)
    
    cv2.imshow("circles", circles_img)
    cv2.waitKey(0)

    # linesP = cv2.HoughLinesP(255 - bil, 2, np.pi / 180, 10, None, 50, 10)

    # if linesP is not None:
    #     for i in range(0, len(linesP)):
    #         l = linesP[i][0]
    #         cv2.line(img, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)

    # cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", img)
    # cv2.waitKey(0)

    cv2.destroyAllWindows() 


dir_str = "component_dataset"


for filename in os.listdir(dir_str):
    dataset_img = cv2.imread(f"{dir_str}/{filename}", cv2.IMREAD_GRAYSCALE)
    result = process_image(dataset_img)
