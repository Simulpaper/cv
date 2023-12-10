import cv2
import re
import sys
import os
import numpy as np


from parse_dataset import get_dataset

def get_component_classifications(orb, t_lower, t_upper, img, dataset, num):
    
    bil = img.copy()
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

    # Applying the Canny Edge filter
    input_edge = cv2.Canny(bil, t_lower, t_upper)

    # cv2.imshow(f"Canny edged", input_edge)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    rows = img.shape[0]
    cols = img.shape[1]
    # print(f"max x: {cols}; max y: {rows}")
    circles = cv2.HoughCircles(input_edge, cv2.HOUGH_GRADIENT, 1, max(rows, cols),
    param1=300, param2=30,
    minRadius=(min(rows, cols) // 4), maxRadius=(int(min(rows, cols) // 2 * 1.2)))
    toCompare = set()
    if circles is not None:
        circles = circles.astype(int)
        circles = circles[0]
        toCompare = set(["voltagesourceu", "voltagesourced", "currentsourceu", "currentsourced", "voltagesourcer", "voltagesourcel", "currentsourcer", "currentsourcel", "lightbulb", "resistor"])
        # print("Circle in component detected!")
        # print(f"detected circle x: {circles[0][0]}, y: {circles[0][1]},  r: {circles[0][2]}")
        # circles_img = input_edge.copy()
        # circles_img = cv2.circle(circles_img, (circles[0][0], circles[0][1]), circles[0][2], (255, 0, 255), 3)
        # cv2.imshow("detected circles", circles_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
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
                # toCompare = set(["resistor", "dioded", "diodeu", "diodel", "dioder", "switch"])
                toCompare = set(["resistor", "switch"])
            else:
                wire = [("wire", 1, 1)]
                print(wire)
                return wire
        else:
            return [("wire", 1, 1)]

    # Now detect the keypoints and compute the descriptors for the query image and train image
    input_keypoints, input_descriptors = orb.detectAndCompute(input_edge,None)
    
    # sift = cv2.SIFT_create()
    # input_keypoints, input_descriptors = sift.detectAndCompute(input_edge,None)

    # fast = cv2.FastFeatureDetector_create()
    # input_keypoints = fast.detect(input_edge, None)

    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    input_keypoints, input_descriptors = brief.compute(input_edge, input_keypoints)
    # print(f"Input image descriptors len: {len(input_descriptors)}, size in bytes: {sys.getsizeof(input_descriptors)}")

    # Initialize the Matcher for matching the keypoints and then match the keypoints
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=False)

    # toCompare = set(["resistor", "dioded", "diodeu", "diodel", "dioder", "switch", "voltagesourceu", "voltagesourced", "currentsourceu", "currentsourced", "voltagesourcer", "voltagesourcel", "currentsourcer", "currentsourcel", "lightbulb"])

    dataset_matches = []
    for filename, edge_img, keypoints, descriptors in dataset:
        if filename not in toCompare:
            continue

        matches = matcher.match(input_descriptors, descriptors)
        matches = sorted(matches, key = lambda x : x.distance)

        num_matches = min(len(matches), num)
        avg_dist = 0
        for i in range(num_matches):
            avg_dist += matches[i].distance
        avg_dist //= num_matches

        # final_img = cv2.drawMatches(input_edge, input_keypoints,
        # edge_img, keypoints, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

        # # Show matches
        # cv2.imshow(f"Matches for {filename}", final_img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        dataset_matches.append((filename, num_matches, avg_dist))
    
    dataset_matches.sort(key = lambda x: x[2])

    # if dataset_matches[0][2] >= 45:
    #     dataset_matches = []
    #     for filename, edge_img, keypoints, descriptors in dataset:

    #         matches = matcher.match(input_descriptors, descriptors)
    #         matches = sorted(matches, key = lambda x : x.distance)
    #         num_matches = min(len(matches), num)
    #         avg_dist = 0
    #         for i in range(num_matches):
    #             avg_dist += matches[i].distance
    #         avg_dist //= num_matches

    #         dataset_matches.append((filename, num_matches, avg_dist))
    
    dataset_matches.sort(key = lambda x: x[2])

    if dataset_matches == []:
        print("NO MATCHES FOUND!!!!")
        return [("wire", 1, 1)]

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

import csv

if __name__ == "__main__":
    # Setting parameter values for Canny
    t_lower = 200 # Lower Threshold
    t_upper = 400 # Upper threshold
    orb = cv2.ORB_create()
    dataset = get_dataset(orb, t_lower, t_upper)

    total = 0
    correct = 0
    correct_orientation = 0
    total_orientation = 0
    with_orientation = set(["voltagesourceu", "voltagesourced", "currentsourceu", "currentsourced", "voltagesourcer", "voltagesourcel", "currentsourcer", "currentsourcel"])
    w_orientation = set(["voltagesource", "currentsource"])

    dir_str = "../generated_components"
    # img = cv2.imread(f"{dir_str}/test04152.jpg", cv2.IMREAD_GRAYSCALE)
    # cla = get_component_classifications(orb, t_lower, t_upper, img, dataset, 100000)

    data = []
    for n in range(450, 451, 10):
        total = 0
        correct = 0
        correct_orientation = 0
        total_orientation = 0
        for filename in os.listdir(dir_str):
            img = cv2.imread(f"{dir_str}/{filename}", cv2.IMREAD_GRAYSCALE)
            classifications = get_component_classifications(orb, t_lower, t_upper, img, dataset, n)
            correct_component = filename[:re.search(r'\d', filename).start()]
            print(f"Supposed to be: {correct_component}")
            print()
            if correct_component in with_orientation:
                if classifications[0][0][:-1] == correct_component[:-1] or (classifications[1][0][:-1] == correct_component[:-1] and classifications[1][2] == classifications[0][2]):
                    correct += 1
                if classifications[0][0] == correct_component or (classifications[1][0] == correct_component and classifications[1][2] == classifications[0][2]):
                    correct_orientation += 1
                total_orientation += 1
            else:
                if classifications[0][0] == correct_component:
                    correct += 1
            total += 1
        data.append({'n': n, 'correct': correct, 'correct_o': correct_orientation})
        print(f"Num: {n}; Correct: {correct}; total: {total}; correct orientation: {correct_orientation}; total orientation: {total_orientation}")
        print()

    # csv_file = 'output.csv'

    # # Open the CSV file in write mode and create a CSV writer
    # with open(csv_file, 'w', newline='') as file:
    #     fieldnames = ['n', 'correct', 'correct_o']
    #     writer = csv.DictWriter(file, fieldnames=fieldnames)

    #     # Write the header
    #     writer.writeheader()

    #     # Write the data for each iteration
    #     for item in data:
    #         writer.writerow(item)
