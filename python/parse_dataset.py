import cv2
import sys
import os
import re
import numpy as np
import random as rng

def get_dataset(orb, t_lower, t_upper):
    dataset_dir = "../component_dataset"
    dataset = []
    dataset_size = 0
    dataset_bytes = 0
    num_descriptors = 0
    for filename in os.scandir(dataset_dir):
        if not filename.is_file():
            continue
        dataset_img = cv2.imread(f"{dataset_dir}/{filename.name}", cv2.IMREAD_GRAYSCALE)
        bil = dataset_img.copy()
        # cv2.imshow('original', bil)
        # cv2.waitKey(0)

        bil = cv2.medianBlur(bil, 3)
        # cv2.imshow('median blurred', bil) 
        # cv2.waitKey(0)

        bil = cv2.fastNlMeansDenoising(bil, None, 30, 7, 11)
        # cv2.imshow('denoise', bil) 
        # cv2.waitKey(0)

        bil = cv2.adaptiveThreshold(bil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                            cv2.THRESH_BINARY, 31, 5) 
        # cv2.imshow('adapt thresh', bil) 
        # cv2.waitKey(0)

        bil = cv2.medianBlur(bil, 5)
        # cv2.imshow('med blurr again', bil) 
        # cv2.waitKey(0)
        
        dataset_edge = cv2.Canny(bil, t_lower, t_upper)
        # cv2.imshow(f"dataset img: {filename.name}", dataset_edge)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows() 
        dataset_keypoints, dataset_descriptors = orb.detectAndCompute(dataset_edge, None)

        # fast = cv2.FastFeatureDetector_create()
        # dataset_keypoints = fast.detect(dataset_edge, None)
        brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
        dataset_keypoints, dataset_descriptors = brief.compute(dataset_edge, dataset_keypoints)
        # if not dataset_keypoints:
        #     continue
        # print(f"Dataset image {filename.name} descriptors len: {len(dataset_descriptors)}, size in bytes: {sys.getsizeof(dataset_descriptors)}")
        dataset.append((filename.name[:re.search(r'\d', filename.name).start()], dataset_edge, dataset_keypoints, dataset_descriptors))
        dataset_size += 1
        dataset_bytes += sys.getsizeof(dataset_descriptors)
        num_descriptors += len(dataset_descriptors)

    # print(f"Dataset number of components: {dataset_size};  avg number of descriptors: {num_descriptors // dataset_size}; avg number of bytes per component: {dataset_bytes // dataset_size}; total bytes: {dataset_bytes}")
    return dataset


