import cv2
import numpy as np

img = cv2.imread("component_images/hough_circuit3a.jpg", cv2.IMREAD_GRAYSCALE) # Read image
if img.shape[1] > img.shape[0]:
    img = cv2.resize(img, (1920, 1080), interpolation=cv2.INTER_AREA)
else:
    img = cv2.resize(img, (1080, 1920), interpolation=cv2.INTER_AREA)
print(f"Height: {img.shape[0]}")
print(f"Width: {img.shape[1]}")
cv2.imshow("original img", img)
cv2.waitKey(0)

threshold_value = 60  # Adjust as needed
binary_img = img.copy()
binary_img = cv2.threshold(binary_img, threshold_value, 255, cv2.THRESH_BINARY)[1]
cv2.imshow("binary img", binary_img)
cv2.waitKey(0)

b_img = binary_img.copy()
b_img = cv2.medianBlur(b_img, 23)
cv2.imshow("blur img", b_img)
cv2.waitKey(0)
 
rows = b_img.shape[0]
cols = b_img.shape[1]
circles = cv2.HoughCircles(b_img, cv2.HOUGH_GRADIENT, 1, min(rows / 8, cols / 8),
param1=500, param2=10,
minRadius=10, maxRadius=45)

if circles is not None:
    circles = circles.astype(int)
    circles = circles[0]
else:
    print("NO CIRCLES DETECTED!!")
    exit

new_circles = []
for i in range(len(circles)):
    new_circles.append((circles[i][0], circles[i][1], circles[i][2]))

circles = new_circles
print(circles)

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
    # neighbor list: [left, right, up, down]
    neighbors = [None] * 4
    this_x = circle[0]
    this_y = circle[1]
    this_r = circle[2]
    dist_differential = 50
    for other_circle in circles:
        if circle != other_circle:
            other_x = other_circle[0]
            other_y = other_circle[1]
            other_r = other_circle[2]
            up_dist = this_y - other_y
            down_dist = other_y - this_y
            left_dist = this_x - other_x
            right_dist = other_x - this_x
            distances = [left_dist, right_dist, up_dist, down_dist]
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

neighbors = {}

for circle in circles:
    nbors = getNeighbors(circle, circles)
    neighbors[circle] = nbors

# print(neighbors)

# for circle in neighbors:
#     x = circle[0]
#     y = circle[1]
#     r = circles[i][2]
#     new_img = img.copy()
#     new_img = cv2.circle(new_img, (x, y), r, (255, 0, 0), 3)
#     cv2.imshow("circle neighbors", new_img)
#     cv2.waitKey(0)
    
#     for neighbor in neighbors[circle]:
#         if neighbor != None:
#             new_img = cv2.circle(new_img, (neighbor[0], neighbor[1]), neighbor[2], (255, 255, 255), 3)

#     cv2.imshow("circle neighbors", new_img)
#     cv2.waitKey(0)

edges = set()
for circle in neighbors:
    for neighbor in neighbors[circle]:
        if neighbor and (circle, neighbor) not in edges and (neighbor, circle) not in edges:
            edges.add((circle, neighbor))
print(edges)

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
    sub_images.append(sub_image)

for i in range(len(sub_images)):
    cv2.imshow(f"component{i}", sub_images[i])
    cv2.waitKey(0)
    cv2.imwrite(f"generated_components/component{i}.jpg", sub_images[i])

