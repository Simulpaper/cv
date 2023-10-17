import cv2
import sys
import os
import re

def get_dataset(orb, bil_params, t_lower, t_upper):
    dataset_dir = "component_dataset"
    dataset = []
    dataset_size = 0
    dataset_bytes = 0
    num_descriptors = 0
    for filename in os.scandir(dataset_dir):
        if not filename.is_file():
            continue
        dataset_img = cv2.imread(f"{dataset_dir}/{filename.name}")
        bil = dataset_img.copy()
        # bil = cv2.bilateralFilter(bil, bil_params[0], bil_params[1], bil_params[2]) 
        dataset_edge = cv2.Canny(bil, t_lower, t_upper)
        dataset_keypoints, dataset_descriptors = orb.detectAndCompute(dataset_edge, None)
        # if not dataset_keypoints:
        #     continue
        # print(f"Dataset image {filename.name} descriptors len: {len(dataset_descriptors)}, size in bytes: {sys.getsizeof(dataset_descriptors)}")
        dataset.append((filename.name[:re.search(r'\d', filename.name).start()], dataset_edge, dataset_keypoints, dataset_descriptors))
        dataset_size += 1
        dataset_bytes += sys.getsizeof(dataset_descriptors)
        num_descriptors += len(dataset_descriptors)

    # print(f"Dataset number of components: {dataset_size};  avg number of descriptors: {num_descriptors // dataset_size}; avg number of bytes per component: {dataset_bytes // dataset_size}; total bytes: {dataset_bytes}")
    return dataset