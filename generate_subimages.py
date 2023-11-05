import cv2
import numpy as np

def load_image(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE) # Read image
    if img.shape[1] > img.shape[0]:
        img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_AREA)
    # print(f"Height: {img.shape[0]}")
    # print(f"Width: {img.shape[1]}")
    # cv2.imshow("original img", img)
    # cv2.waitKey(0)
    return img

def apply_threshold(img, threshold_value=60):
    binary_img = img.copy()
    binary_img = cv2.threshold(binary_img, threshold_value, 255, cv2.THRESH_BINARY)[1]
    # cv2.imshow("binary img", binary_img)
    # cv2.waitKey(0)
    return binary_img

def apply_median_blur(img, ksize=23):
    b_img = img.copy()
    b_img = cv2.medianBlur(b_img, ksize)
    # cv2.imshow("blur img", b_img)
    # cv2.waitKey(0)
    return b_img

def get_circles(img):
    rows = img.shape[0]
    cols = img.shape[1]
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, min(rows / 8, cols / 8),
    param1=500, param2=10,
    minRadius=15, maxRadius=60)

    if circles is not None:
        circles = circles.astype(int)
        circles = circles[0]
    else:
        print("NO CIRCLES DETECTED!!")
        exit

    new_circles = []
    for i in range(len(circles)):
        new_circles.append((circles[i][0], circles[i][1], circles[i][2]))

    print(new_circles)
    return new_circles

def show_circles(img, circles):
    circles_img = img.copy()
    # for x, y, r in circles:
    for i in range(len(circles)):
        x = circles[i][0]
        y = circles[i][1]
        r = circles[i][2]
        # circle center
        circles_img = cv2.circle(circles_img, (x, y), 1, (0, 100, 100), 3)
        circles_img = cv2.circle(circles_img, (x, y), r, (255, 0, 255), 3)


    cv2.imshow("detected circles", circles_img)
    cv2.waitKey(0)

def distance(circle1, circle2):
    x1, y1, _ = circle1
    x2, y2, _ = circle2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

def getNeighbors(circle, circles):
    # direction = ["left", "right", "down", "up"]
    neighbors = [None] * 4
    this_x = circle[0]
    this_y = circle[1]
    this_r = circle[2]
    dist_differential = 50
    for other_circle in circles:
        if circle == other_circle:
            continue
        other_x = other_circle[0]
        other_y = other_circle[1]
        other_r = other_circle[2]
        up_dist = this_y - other_y
        down_dist = other_y - this_y
        left_dist = this_x - other_x
        right_dist = other_x - this_x
        distances = [left_dist, right_dist, down_dist, up_dist]
        e_dist = distance(circle, other_circle)
        for i in range(len(distances)):
            dir_dist = distances[i]
            if dir_dist > 0 and e_dist - dir_dist <= dist_differential and (neighbors[i] == None or dir_dist < neighbors[i][1]):
                neighbors[i] = (other_circle, dir_dist)
    
    result = [None] * 4
    for i in range(len(result)):
        if neighbors[i] != None:
            result[i] = neighbors[i][0]
    
    return result

def show_neighbors(img, neighbors):
    print(neighbors)
    for circle in neighbors:
        x = circle[0]
        y = circle[1]
        r = circle[2]
        new_img = img.copy()
        new_img = cv2.circle(new_img, (x, y), r, (255, 0, 0), 3)
        cv2.imshow("circle neighbors", new_img)
        cv2.waitKey(0)
        
        for neighbor in neighbors[circle]:
            if neighbor != None:
                new_img = cv2.circle(new_img, (neighbor[0], neighbor[1]), neighbor[2], (255, 255, 255), 3)

        cv2.imshow("circle neighbors", new_img)
        cv2.waitKey(0)

# def get_nodes(circles, neighbors):
#     node_map = {}
#     node_index = 0

#     # want bottom left most node to be node 0
#     # find bottom right most node and then just traverse as far right as possible
#     bot_right_circle = circles[0]
#     for circle in circles:
#         # bottom right circle is the one with the greatest sum of coordinates
#         if circle[0] + circle[1] >= bot_right_circle[0] + bot_right_circle[1]:
#             bot_right_circle = circle

#     # bottom left node is traversing left all the way from the bottom right node
#     bot_left_circle = bot_right_circle
#     while neighbors[bot_left_circle][0] != None:
#         bot_left_circle = neighbors[bot_left_circle][0]
    
#     node_map[bot_left_circle] = node_index
#     node_index += 1
#     for circle in circles:
#         if circle not in node_map:
#             node_map[circle] = node_index
#             node_index += 1
    
#     return node_map

def get_subimages(img, edges):
    sub_images = []

    shave_off = 50
    add_size = 90 

    for circle1, circle2 in edges:

        # if component is horizontal
        if abs(circle1[0] - circle2[0]) > abs(circle1[1] - circle2[1]):
            x_min = min(circle1[0], circle2[0]) + shave_off
            x_max = max(circle1[0], circle2[0]) - shave_off
            y_min = max(min(circle1[1], circle2[1]) - add_size, 0)
            y_max = min(max(circle1[1], circle2[1]) + add_size, img.shape[0])
        # else component is vertical
        else:
            x_min = max(min(circle1[0], circle2[0]) - add_size, 0)
            x_max = min(max(circle1[0], circle2[0]) + add_size, img.shape[1])
            y_min = min(circle1[1], circle2[1]) + shave_off
            y_max = max(circle1[1], circle2[1]) - shave_off

        sub_image = img[y_min:y_max, x_min:x_max]
        # first node, second node, subimage
        sub_images.append([circle1, circle2, sub_image])
    return sub_images

def get_edges_subimages(filename):
    user_img = load_image(filename)
    thresholded_img = apply_threshold(user_img)
    med_blurred_img = apply_median_blur(thresholded_img)
    circles = get_circles(med_blurred_img)
    show_circles(user_img, circles)
    
    neighbors = {}
    for circle in circles:
        nbors = getNeighbors(circle, circles)
        neighbors[circle] = nbors
    print(neighbors)

    # show_neighbors(user_img, neighbors)

    # node_map = get_nodes(circles, neighbors)

    edges = set()
    for circle in neighbors:
        circle_neighbors = neighbors[circle]
        for i in range(len(circle_neighbors)):
            neighbor = circle_neighbors[i]
            if neighbor and (circle, neighbor) not in edges and (neighbor, circle) not in edges:
                # if neighbor is to the left or below
                if i % 2 == 0:
                    edges.add((neighbor, circle))
                # neighbor is to the right or above
                else:
                    edges.add((circle, neighbor))
    print(edges)

    sub_images = get_subimages(user_img, edges)

    # for i in range(len(sub_images)):
    #     cv2.imshow(f"component{i}", sub_images[i][2])
    #     cv2.waitKey(0)
    #     cv2.imwrite(f"generated_components/component{i}.jpg", sub_images[i][2])
    return sub_images

if __name__ == "__main__":
    get_edges_subimages("component_images/demo.jpg")