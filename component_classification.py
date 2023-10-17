import cv2
import re
import sys

from parse_dataset import get_dataset

def get_component_classifications(orb, bil_params, t_lower, t_upper, img, dataset):
    
    bil = img.copy()
    # bil = cv2.bilateralFilter(bil,bil_params[0], bil_params[1], bil_params[2]) 
    # cv2.imshow('Bilateral Filtering', bil) 
    # cv2.waitKey(0)

    # Applying the Canny Edge filter
    input_edge = cv2.Canny(bil, t_lower, t_upper)

    cv2.imshow(f"Canny edged", input_edge)
    cv2.waitKey(0)

    # Now detect the keypoints and compute the descriptors for the query image and train image
    input_keypoints, input_descriptors = orb.detectAndCompute(input_edge,None)
    # print(f"Input image descriptors len: {len(input_descriptors)}, size in bytes: {sys.getsizeof(input_descriptors)}")

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
    # print(dataset_matches)

    top_three_matches = []
    have = set()
    for component_type, num_matches, avg_dist in dataset_matches:
        if component_type not in have:
            have.add(component_type)
            top_three_matches.append((component_type, num_matches, avg_dist))
        if len(top_three_matches) == 3:
            break

    print(top_three_matches)
    return top_three_matches

    # matches = list of DMatch objects
    # DMatch object
        # DMatch.distance - Distance between descriptors. The lower, the better it is.
        # DMatch.trainIdx - Index of the descriptor in train descriptors
        # DMatch.queryIdx - Index of the descriptor in query descriptors
        # DMatch.imgIdx - Index of the train image.



if __name__ == "__main__":
    img = cv2.imread(f"generated_components/component3.jpg") # Read image
    bil_params = (5, 200, 200)
    # Setting parameter values for Canny
    t_lower = 200 # Lower Threshold
    t_upper = 500 # Upper threshold
    orb = cv2.ORB_create()
    dataset = get_dataset(orb, bil_params, t_lower, t_upper)
    classifications = get_component_classifications(orb, bil_params, t_lower, t_upper, img, dataset)
