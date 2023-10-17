import cv2

train_img = cv2.imread("component_images/train1.jpg") # Read image
train_img = cv2.resize(train_img, (900,800))
query_img = cv2.imread("component_images/train2.jpg") # Read image
query_img = cv2.resize(query_img, (900,800))

# Setting parameter values
t_lower = 200 # Lower Threshold
t_upper = 500 # Upper threshold

# Applying the Canny Edge filter
train_edge = cv2.Canny(train_img, t_lower, t_upper)
query_edge = cv2.Canny(query_img, t_lower, t_upper)

# cv2.imshow('train original', train_img)
# cv2.imshow('train edge', train_edge)
# cv2.imshow('query original', query_img)
# cv2.imshow('query edge', query_edge)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# # Convert it to grayscale
# query_img_bw = cv2.cvtColor(query_img,cv2.COLOR_BGR2GRAY)
# train_img_bw = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

# # Initialize the ORB detector algorithm
orb = cv2.ORB_create()

# Initiate SIFT detector
sift = cv2.SIFT_create()

# # Now detect the keypoints and compute
# # the descriptors for the query image
# # and train image
queryKeypoints, queryDescriptors = orb.detectAndCompute(query_edge,None)
print("ORB query keypoints len: " + str(len(queryKeypoints)))

trainKeypoints, trainDescriptors = orb.detectAndCompute(train_edge,None)
print("ORB train keypoints len: " + str(len(trainKeypoints)))

# find the keypoints and descriptors with SIFT
kp1, des1 = sift.detectAndCompute(query_edge,None)
print("SIFT query keypoints len: " + str(len(kp1)))

kp2, des2 = sift.detectAndCompute(train_edge,None)
print("SIFT train keypoints len: " + str(len(kp2)))


# BFMatcher with default params
bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k = 2)
# Apply ratio test
good = []
for m,n in matches:
    if m.distance < 0.75*n.distance:
        good.append(m)

matches = sorted(good, key = lambda x : x.distance)
# cv.drawMatchesKnn expects list of lists as matches.
# img3 = cv2.drawMatchesKnn(query_edge,kp1,train_edge,kp2,good,None,flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
print(f"Number of matches with SIFT: {len(matches)}")
avg_dist = 0
for i in range(len(matches)):
    avg_dist += matches[i].distance
if len(matches) != 0:
    avg_dist //= len(matches)
print(f"Average distance with SIFT: {avg_dist}")
# img3 = cv2.drawMatches(query_edge, kp1,
# train_edge, kp2, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
# img3 = cv2.resize(img3, (1000,650))
# cv2.imshow("Matches", img3)
# cv2.waitKey(0)

# # Initialize the Matcher for matching
# # the keypoints and then match the
# # keypoints
matcher = cv2.BFMatcher(cv2.NORM_HAMMING,crossCheck=True)
matches = matcher.match(queryDescriptors,trainDescriptors)


# matches = list of DMatch objects
# DMatch object
    # DMatch.distance - Distance between descriptors. The lower, the better it is.
    # DMatch.trainIdx - Index of the descriptor in train descriptors
    # DMatch.queryIdx - Index of the descriptor in query descriptors
    # DMatch.imgIdx - Index of the train image.

# The matches with shorter distance are the ones we want.
matches = sorted(matches, key = lambda x : x.distance)
print(f"Number of matches with ORB: {len(matches)}")
avg_dist = 0
for i in range(len(matches)):
    avg_dist += matches[i].distance
avg_dist //= len(matches)
print(f"Average distance with ORB: {avg_dist}")

# # draw the matches to the final image
# # containing both the images the drawMatches()
# # function takes both images and keypoints
# # and outputs the matched query image with
# # its train image
final_img = cv2.drawMatches(query_edge, queryKeypoints,
train_edge, trainKeypoints, matches[:20], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

final_img = cv2.resize(final_img, (1000,650))

# Show the final image
cv2.imshow("Matches", final_img)
cv2.waitKey(0)