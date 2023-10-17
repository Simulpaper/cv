import cv2
import os
import re
import sys

dataset_dir = "component_dataset"
input_dir = "generated_components"

# Setting parameter values for Canny
t_lower = 200 # Lower Threshold
t_upper = 500 # Upper threshold

# Initialize the ORB detector algorithm
orb = cv2.ORB_create()
sift = cv2.SIFT_create()

dataset = []

dataset_size = 0
dataset_bytes = 0
num_descriptors = 0
for filename in os.scandir(dataset_dir):
    if not filename.is_file():
        continue
    dataset_img = cv2.imread(f"{dataset_dir}/{filename.name}")
    
    dataset_edge = cv2.Canny(dataset_img, t_lower, t_upper)
    dataset_keypoints, dataset_descriptors = orb.detectAndCompute(dataset_edge, None)
    # dataset_keypoints, dataset_descriptors = sift.detectAndCompute(dataset_edge,None)
    # if not dataset_keypoints:
    #     continue
    print(f"Dataset image {filename.name} descriptors len: {len(dataset_descriptors)}, size in bytes: {sys.getsizeof(dataset_descriptors)}")
    dataset.append((filename.name, dataset_edge, dataset_keypoints, dataset_descriptors))
    dataset_size += 1
    dataset_bytes += sys.getsizeof(dataset_descriptors)
    num_descriptors += len(dataset_descriptors)

print(f"Dataset number of components: {dataset_size};  avg number of descriptors: {num_descriptors // dataset_size}; avg number of bytes per component: {dataset_bytes // dataset_size}; total bytes: {dataset_bytes}")

input_img_name = "component3.jpg"

input_img = cv2.imread(f"{input_dir}/{input_img_name}") # Read image

bil = input_img.copy()
bil = cv2.bilateralFilter(bil,5,200,200) 
cv2.imshow('Bilateral Filtering', bil) 
cv2.waitKey(0)

# Applying the Canny Edge filter
input_edge = cv2.Canny(bil, t_lower, t_upper)

cv2.imshow(f"Canny edged", input_edge)
cv2.waitKey(0)

# Now detect the keypoints and compute the descriptors for the query image and train image
input_keypoints, input_descriptors = orb.detectAndCompute(input_edge,None)
print(f"Input image descriptors len: {len(input_descriptors)}, size in bytes: {sys.getsizeof(input_descriptors)}")


# Initialize the Matcher for matching the keypoints and then match the keypoints
matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

dataset_matches = []
# bestMatch = (filename, num_matches, avg_dist)
bestMatch = None
for filename, edge_img, keypoints, descriptors in dataset:
    matches = matcher.match(input_descriptors, descriptors)
    # matches = sorted(matches, key = lambda x : x.distance)
    num_matches = len(matches)
    avg_dist = 0
    for i in range(len(matches)):
        avg_dist += matches[i].distance
    avg_dist //= len(matches)

    final_img = cv2.drawMatches(input_edge, input_keypoints,
    edge_img, keypoints, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Show matches
    # cv2.imshow(f"Matches for {filename}", final_img)
    # cv2.waitKey(0)
    
    dataset_matches.append((filename, num_matches, avg_dist))
    
dataset_matches.sort(key = lambda x: (-x[1], x[2]))
print(dataset_matches)

top_three_matches = {}
for filename, num_matches, avg_dist in dataset_matches:
    component_type = filename[:re.search(r'\d', filename).start()]
    if component_type not in top_three_matches:
        top_three_matches[component_type] = (filename, num_matches, avg_dist)
    if len(top_three_matches) == 3:
        break

print("\n")
print(f"{input_img_name} best match: {dataset_matches[0][0]} with avg distance {dataset_matches[0][2]}")
print("\n")

#print(top_three_matches)

# matches = list of DMatch objects
# DMatch object
    # DMatch.distance - Distance between descriptors. The lower, the better it is.
    # DMatch.trainIdx - Index of the descriptor in train descriptors
    # DMatch.queryIdx - Index of the descriptor in query descriptors
    # DMatch.imgIdx - Index of the train image.


print("Generated netlist:")
print("R1   N1  N2")
print("R2   N2  N0")
print("V1   N0  N1")
print("\n")
