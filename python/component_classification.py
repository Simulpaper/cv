import cv2
import re
import sys
import numpy as np

from parse_dataset import get_dataset

def get_component_classifications(orb, t_lower, t_upper, img, dataset):
    
    bil = img.copy()
    bil = cv2.medianBlur(bil, 3)
    cv2.imshow('median blurred', bil) 
    cv2.waitKey(0)

    bil = cv2.fastNlMeansDenoising(bil, None, 30, 11, 41)
    cv2.imshow('denoise', bil) 
    cv2.waitKey(0)


    bil = cv2.adaptiveThreshold(bil, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                        cv2.THRESH_BINARY, 199, 5) 
    cv2.imshow('adapt thresh', bil) 
    cv2.waitKey(0)

    bil = cv2.medianBlur(bil, 5)
    cv2.imshow('med blurr again', bil) 
    cv2.waitKey(0)

    # bil = cv2.fastNlMeansDenoising(bil, None, 30, 11, 21)
    # cv2.imshow('denoise', bil) 
    # cv2.waitKey(0)

    # bil = cv2.medianBlur(bil, 3)
    # cv2.imshow('med blurr again', bil) 
    # cv2.waitKey(0)

    # Applying the Canny Edge filter
    input_edge = cv2.Canny(bil, t_lower, t_upper)

    cv2.imshow(f"Canny edged", input_edge)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    rows = img.shape[0]
    cols = img.shape[1]
    circles = cv2.HoughCircles(input_edge, cv2.HOUGH_GRADIENT, 1, max(rows, cols),
    param1=300, param2=40,
    minRadius=(min(rows, cols) // 6), maxRadius=(min(rows, cols)))
    toCompare = set()
    if circles is not None:
        circles = circles.astype(int)
        circles = circles[0]
        toCompare = set(["voltage_source+", "voltage_source-", "current_source+", "current_source-", "lightbulb"])
        print("Circle in component detected!")
        circles_img = input_edge.copy()
        circles_img = cv2.circle(circles_img, (circles[0][0], circles[0][1]), circles[0][2], (255, 0, 255), 3)
        cv2.imshow("detected circles", circles_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        linesP = cv2.HoughLinesP(255 - bil, 2, np.pi / 180, 50, None, 50, 10)
        isWire = True
        lowX = bil.shape[1]
        highX = 0
        lowY = bil.shape[0]
        highY = 0
        if linesP is not None:
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
            # if is vertical image and difference in Xs is a lot OR is horizontal image and difference in Ys is a lot
            if (bil.shape[0] > bil.shape[1] and highX - lowX > bil.shape[1] // 3)  or (bil.shape[1] > bil.shape[0] and highY - lowY > bil.shape[0] // 3):
                toCompare = set(["resistor", "diode", "switch"])
            else:
                return [("wire", 1, 1)]
        else:
            toCompare = set(["wire", "resistor", "diode", "switch"])

    # Now detect the keypoints and compute the descriptors for the query image and train image
    input_keypoints, input_descriptors = orb.detectAndCompute(input_edge,None)
    # print(f"Input image descriptors len: {len(input_descriptors)}, size in bytes: {sys.getsizeof(input_descriptors)}")

    # Initialize the Matcher for matching the keypoints and then match the keypoints
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

    dataset_matches = []
    # bestMatch = (filename, num_matches, avg_dist)
    bestMatch = None
    for filename, edge_img, keypoints, descriptors in dataset:
        if filename not in toCompare:
            continue

        # matches = matcher.knnMatch(input_descriptors, descriptors,k=2)
        # # Apply ratio test
        # good = []
        # good1 = []
        # for m,n in matches:
        #     if m.distance < 0.7*n.distance:
        #         good.append(m)
        #         good1.append([m])
        # num_matches = len(good)
        # if num_matches == 0:
        #     continue
        # avg_dist = 0
        # for i in range(len(good)):
        #     avg_dist += good[i].distance
        # avg_dist //= len(good)

        matches = matcher.match(input_descriptors, descriptors)
        # matches = sorted(matches, key = lambda x : x.distance)
        num_matches = len(matches)
        avg_dist = 0
        for i in range(len(matches)):
            avg_dist += matches[i].distance
        avg_dist //= len(matches)

        # final_img = cv2.drawMatches(input_edge, input_keypoints,
        # edge_img, keypoints, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # img3 = cv2.drawMatchesKnn(input_edge,input_keypoints,edge_img,keypoints,good1,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        # cv2.imshow(f"Matches for {filename}", img3)
        # cv2.waitKey(0)
        # Show matches
        # cv2.imshow(f"Matches for {filename}", final_img)
        # cv2.waitKey(0)
        dataset_matches.append((filename, num_matches, avg_dist))
        
    dataset_matches.sort(key = lambda x: x[2])
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
    img = cv2.imread(f"../generated_components/component2.jpg", cv2.IMREAD_GRAYSCALE) # Read image
    # Setting parameter values for Canny
    t_lower = 200 # Lower Threshold
    t_upper = 400 # Upper threshold
    orb = cv2.ORB_create()
    dataset = get_dataset(orb, t_lower, t_upper)
    classifications = get_component_classifications(orb, t_lower, t_upper, img, dataset)
