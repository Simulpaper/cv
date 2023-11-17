import cv2
import sys
import os
import re
import numpy as np
import random as rng

def make_dataset(orb, t_lower, t_upper):
    dataset_dir = "component_dataset"
    dataset = []
    dataset_size = 0
    dataset_bytes = 0
    num_descriptors = 0

    with open("dataset_info.txt", "w") as file:
        for filename in os.scandir(dataset_dir):
            if not filename.is_file():
                continue
            dataset_img = cv2.imread(f"{dataset_dir}/{filename.name}", cv2.IMREAD_GRAYSCALE)
            bil = dataset_img.copy()

            bil = cv2.medianBlur(bil, 3)

            bil = cv2.fastNlMeansDenoising(bil, None, 30, 11, 41)

            bil = cv2.adaptiveThreshold(bil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                cv2.THRESH_BINARY, 199, 5) 

            bil = cv2.medianBlur(bil, 5)
            
            dataset_edge = cv2.Canny(bil, t_lower, t_upper)
            cv2.imshow(f"dataset img: {filename.name}", dataset_edge)
            cv2.waitKey(0)
            cv2.destroyAllWindows() 
            dataset_keypoints, dataset_descriptors = orb.detectAndCompute(dataset_edge, None)
            # if not dataset_keypoints:
            #     continue

            component_type = filename.name[:re.search(r'\d', filename.name).start()]

            print(sys.getsizeof(dataset_descriptors))
            
            file.write(f"{component_type}: {dataset_descriptors}\n")

if __name__ == "__main__":
    t_lower = 200 # Lower Threshold
    t_upper = 400 # Upper threshold
    orb = cv2.ORB_create()
    make_dataset(orb, t_lower, t_upper)