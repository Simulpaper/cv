import cv2
import numpy as np
import os

def process_image(img):
    bil = cv2.medianBlur(img, 3)
    # cv2.imshow('median blurred', bil) 
    # cv2.waitKey(0)

    bil = cv2.fastNlMeansDenoising(bil, None, 30, 11, 21)
    # cv2.imshow('denoise', bil) 
    # cv2.waitKey(0)

    bil = cv2.adaptiveThreshold(bil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY, 191, 5) 
    # cv2.imshow('adapt thresh', bil) 
    # cv2.waitKey(0)

    bil = cv2.medianBlur(bil, 5)
    # cv2.imshow('med blurr again', bil) 
    # cv2.waitKey(0)

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
        circles_img = bil
        # for i in range(len(circles)):
        #     circles_img = cv2.circle(circles_img, (circles[i][0], circles[i][1]), circles[i][2], (0, 100, 100), 3)
        
        # cv2.imshow("circles", circles_img)
        # cv2.waitKey(0)
    else:
        print("NO CIRCLES DETECTED!!")
        linesP = cv2.HoughLinesP(255 - bil, 2, np.pi / 180, 50, None, 50, 10)
        blankImg = np.zeros((bil.shape[0], bil.shape[1]), dtype = np.uint8)
        if linesP is not None:
            isWire = True
            lowX = bil.shape[1]
            highX = 0
            lowY = bil.shape[0]
            highY = 0
            for i in range(len(linesP)):
                l = linesP[i][0]
                lowX = min(lowX, l[0])
                lowX = min(lowX, l[2])
                lowY = min(lowY, l[1])
                lowY = min(lowY, l[3])
                highX = max(highX, l[0])
                highX = max(highX, l[2])
                highY = max(highY, l[1])
                highY = max(highY, l[3])
                cv2.line(blankImg, (l[0], l[1]), (l[2], l[3]), 255, 1, cv2.LINE_AA)

        # if vertical image and 
        if (bil.shape[0] > bil.shape[1] and highX - lowX > bil.shape[1] // 3)  or (bil.shape[1] > bil.shape[0] and highY - lowY > bil.shape[0] // 3):
            isWire = False
        print(f"Is a wire: {isWire}")
        cv2.imshow("Detected Lines (in red) - Probabilistic Line Transform", blankImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows() 
    


dir_str = "component_dataset"


for filename in os.listdir(dir_str):
    dataset_img = cv2.imread(f"{dir_str}/{filename}", cv2.IMREAD_GRAYSCALE)
    result = process_image(dataset_img)
