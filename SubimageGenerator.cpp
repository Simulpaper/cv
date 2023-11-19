#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <tuple>
#include "SubimageGenerator.hpp"
#include "HelperStructs.hpp"


cv::Mat loadImage(const std::string& filename) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (img.cols > img.rows) {
        cv::resize(img, img, cv::Size(1920, 1080), cv::INTER_AREA);
    }
    else {
        cv::resize(img, img, cv::Size(1080, 1920), cv::INTER_AREA);
    }
    // cv::imshow("Grayscaled img", img);
    // cv::waitKey(0);
    return img;
}

cv::Mat applyThreshold(const cv::Mat& img) {
    cv::Mat binaryImg = img.clone();
    cv::fastNlMeansDenoising(binaryImg, binaryImg, 30, 21, 11);
    cv::adaptiveThreshold(binaryImg, binaryImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 49, 5);

    return binaryImg; 
}

cv::Mat applyMedianBlur(const cv::Mat& img, int ksize) {
    cv::Mat blurredImg = img.clone();
    cv::medianBlur(blurredImg, blurredImg, ksize);
    // cv::imshow("Median-blurred img", blurredImg);
    // cv::waitKey(0);
    return blurredImg;
}

std::vector<cv::Vec3i> getCircles(const cv::Mat& img) {
    int rows = img.rows;
    int cols = img.cols;
    std::vector<cv::Vec3f> fCircles;
    cv::HoughCircles(img, fCircles, cv::HOUGH_GRADIENT, 1, std::min(rows / 8, cols / 8), 200, 10, 10, 80);

    
    if (fCircles.empty()) {
        std::cout << "NO CIRCLES DETECTED!!" << std::endl;
        exit(EXIT_FAILURE);
    }
    std::vector<cv::Vec3i> circles;

    for (const cv::Vec3f& fCircle : fCircles) {
        int x = static_cast<int>(std::round(fCircle[0]));
        int y = static_cast<int>(std::round(fCircle[1]));
        int r = static_cast<int>(std::round(fCircle[2]));
        circles.push_back(cv::Vec3i(x, y, r));
    }

    return circles;
}

void showCircles(const cv::Mat& img, const std::vector<cv::Vec3i>& circles) {
    cv::Mat circlesImg = img.clone();
    for (const cv::Vec3i& circle : circles) {
        int x = circle[0];
        int y = circle[1];
        int r = circle[2];
        cv::circle(circlesImg, cv::Point(x, y), 1, cv::Scalar(0, 100, 100), 3);
        cv::circle(circlesImg, cv::Point(x, y), r, cv::Scalar(255, 0, 255), 3);
    }
    cv::imshow("Detected Circles", circlesImg);
    cv::waitKey(0);
}

double distance(const cv::Vec3i& circle1, const cv::Vec3i& circle2) {
    int x1 = circle1[0];
    int y1 = circle1[1];
    int x2 = circle2[0];
    int y2 = circle2[1];
    return std::sqrt(std::pow(x2 - x1, 2) + std::pow(y2 - y1, 2));
}

std::vector<cv::Vec3i> getNeighbors(const cv::Vec3i& circle, const std::vector<cv::Vec3i>& circles) {
    std::vector<cv::Vec3i> neighbors(4, cv::Vec3i(0, 0, -1));
    std::vector<double> bestDistances(4, 0);
    int thisX = circle[0];
    int thisY = circle[1];
    int thisR = circle[2];
    double distDifferential = 50; // a right triangle of dirDist and 50 pixels as base and height should be the max to be considered a neighbor
    double dirDist;

    for (const cv::Vec3i& otherCircle : circles) {
        if (circle == otherCircle) {
            continue;
        }
        int otherX = otherCircle[0];
        int otherY = otherCircle[1];
        int otherR = otherCircle[2];
        double upDist = 0.0 + thisY - otherY;
        double downDist = 0.0 + otherY - thisY;
        double leftDist = 0.0 + thisX - otherX;
        double rightDist = 0.0 + otherX - thisX;
        std::vector<double> distances = {leftDist, rightDist, downDist, upDist};
        double eDist = distance(circle, otherCircle);

        for (int i = 0; i < 4; i++) {
            dirDist = distances[i];
            if (dirDist > 0 && eDist <= std::sqrt(std::pow(dirDist, 2) + std::pow(distDifferential, 2)) && (neighbors[i][2] == -1 || dirDist < bestDistances[i])) {
                neighbors[i] = otherCircle;
                bestDistances[i] = dirDist;
            }
        }
    }

    return neighbors;
}

void showNeighbors(const cv::Mat& img, const std::map<cv::Vec3i, std::vector<cv::Vec3i>, Vec3iCompare>& neighbors) {
    for (const auto& entry : neighbors) {
        const cv::Vec3i& circle = entry.first;
        const std::vector<cv::Vec3i>& circle_neighbors = entry.second;
        
        // Create a copy of the original image
        cv::Mat new_img = img.clone();

        // Draw the circle in red
        cv::circle(new_img, cv::Point(circle[0], circle[1]), circle[2], cv::Scalar(128, 128, 128), 3);

        // Draw the neighbors in white
        for (const cv::Vec3i& neighbor : circle_neighbors) {
            if (neighbor[2] != -1) {
                cv::circle(new_img, cv::Point(neighbor[0], neighbor[1]), neighbor[2], cv::Scalar(255, 255, 255), 3);
            }
        }

        // Show the image
        cv::imshow("Circle Neighbors", new_img);
        cv::waitKey(0);
    }
}

std::vector<std::pair<cv::Vec3i, cv::Vec3i>> getEdges(const std::map<cv::Vec3i, std::vector<cv::Vec3i>, Vec3iCompare>& neighbors) {
    std::vector<std::pair<cv::Vec3i, cv::Vec3i>> edges;

    for (const auto& entry : neighbors) {
        const cv::Vec3i& circle = entry.first;
        const std::vector<cv::Vec3i>& circle_neighbors = entry.second;

        for (int i = 0; i < circle_neighbors.size(); ++i) {
            const cv::Vec3i& neighbor = circle_neighbors[i];

            if (neighbor[2] == -1) {
                continue;
            }

            bool isDuplicate = false;
            for (const std::pair<cv::Vec3i, cv::Vec3i>& edge : edges) {
                if ((edge.first == neighbor && edge.second == circle) ||
                    (edge.second == neighbor && edge.first == circle)) {
                    isDuplicate = true;
                    break;
                }
            }

            if (!isDuplicate) {
                // put left and down neighbors as the first node in the edge pair
                if (i % 2 == 0) {
                    edges.push_back(std::make_pair(neighbor, circle));
                } else {
                    edges.push_back(std::make_pair(circle, neighbor));
                }
            }
        }
    }

    // Print the 'edges'
    for (const auto& edge : edges) {
        std::cout << "(" << edge.first[0] << ", " << edge.first[1] << ", " << edge.first[2] << ")"
                  << " <-> "
                  << "(" << edge.second[0] << ", " << edge.second[1] << ", " << edge.second[2] << ")"
                  << std::endl;
    }

    return edges;
}

std::vector<ComponentSubimage> getSubimages(const cv::Mat& userImg, const std::vector<std::pair<cv::Vec3i, cv::Vec3i>>& edges) {
    std::vector<ComponentSubimage> subImages;

    int shaveOff = 50; // how much to shave from sides of components connected to nodes
    int jumpSize = 2;
    int initialWidth = 30;
    int extraWidth = 20;

    cv::Mat thresholdImg = userImg.clone();
    cv::fastNlMeansDenoising(thresholdImg, thresholdImg, 50, 11, 11);
    cv::adaptiveThreshold(thresholdImg, thresholdImg, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 27, 5);

    for (const auto& edge : edges) {
        const cv::Vec3i& circle1 = edge.first;
        const cv::Vec3i& circle2 = edge.second;

        bool isHorizontal = std::abs(circle1[0] - circle2[0]) > std::abs(circle1[1] - circle2[1]);
        int leftX, rightX, lowerY, upperY;
        bool hasBlack;

        if (isHorizontal) {
            leftX = std::min(circle1[0], circle2[0]) + shaveOff;
            rightX = std::max(circle1[0], circle2[0]) - shaveOff;

            upperY = std::min(circle1[1], circle2[1]) + initialWidth;
            while (upperY < thresholdImg.rows) {
                hasBlack = false;
                for (int x = leftX; x < rightX; x++) {
                    if (thresholdImg.at<uchar>(upperY, x) < 255) {
                        hasBlack = true;
                        break;
                    }
                }
                if (hasBlack) {
                    upperY += jumpSize;
                } else {
                    break;
                }
            }
            upperY = std::min(upperY + extraWidth, thresholdImg.rows);

            lowerY = std::min(circle1[1], circle2[1]) - initialWidth;
            while (lowerY > 0) {
                hasBlack = false;
                for (int x = leftX; x < rightX; x++) {
                    if (thresholdImg.at<uchar>(lowerY, x) < 255) {
                        hasBlack = true;
                        break;
                    }
                }
                if (hasBlack) {
                    lowerY -= jumpSize;
                } else {
                    break;
                }
            }
            lowerY = std::max(lowerY - extraWidth, 0);
        } else {
            lowerY = std::min(circle1[1], circle2[1]) + shaveOff;
            upperY = std::max(circle1[1], circle2[1]) - shaveOff;

            rightX = std::min(circle1[0], circle2[0]) + initialWidth;
            while (rightX < thresholdImg.cols) {
                hasBlack = false;
                for (int y = lowerY; y < upperY; y++) {
                    if (thresholdImg.at<uchar>(y, rightX) < 255) {
                        hasBlack = true;
                        break;
                    }
                }
                if (hasBlack) {
                    rightX += jumpSize;
                } else {
                    break;
                }
            }
            rightX = std::min(rightX + extraWidth, thresholdImg.cols);

            leftX = std::min(circle1[0], circle2[0]) - initialWidth;
            while (leftX > 0) {
                hasBlack = false;
                for (int y = lowerY; y < upperY; y++) {
                    if (thresholdImg.at<uchar>(y, leftX) < 255) {
                        hasBlack = true;
                        break;
                    }
                }
                if (hasBlack) {
                    leftX -= jumpSize;
                } else {
                    break;
                }
            }
            leftX = std::max(leftX - extraWidth, 0);
        }
        // std::cout << rightX - leftX << " x " << upperY - lowerY << std::endl;
        cv::Rect roi(leftX, lowerY, rightX - leftX, upperY - lowerY);
        cv::Mat subimage = userImg(roi).clone();

        // Store the subimage along with the associated circles
        ComponentSubimage cimage;
        cimage.node1 = circle1;
        cimage.node2 = circle2;
        cimage.image = subimage;
        subImages.push_back(cimage);
    }

    return subImages;
}

std::vector<ComponentSubimage> SubimageGenerator::generateSubimages(const std::string& filename) {
    cv::Mat userImg = loadImage(filename);
    std::cout << "Loaded image" << std::endl;
    cv::Mat thresholdedImg = applyThreshold(userImg, 80);
    std::cout << "Applied threshold" << std::endl;
    cv::Mat medBlurredImg = applyMedianBlur(thresholdedImg, 31);
    std::cout << "Applied median blur" << std::endl;
    std::vector<cv::Vec3i> circles = getCircles(medBlurredImg);
    std::cout << "got circles" << std::endl;
    // showCircles(userImg, circles);
    
    // get neighbors:
    std::map<cv::Vec3i, std::vector<cv::Vec3i>, Vec3iCompare> neighbors;
    int neighborCount;
    for (const cv::Vec3i& circle : circles) {
        std::vector<cv::Vec3i> circle_neighbors = getNeighbors(circle, circles);
        neighborCount = 0;
        for (const cv::Vec3i& neighbor : circle_neighbors) {
            if (neighbor[2] != -1) {
                neighborCount += 1;
            }
        }
        if (neighborCount >= 2) {
            neighbors[circle] = circle_neighbors;
        }
    }
    

    // showNeighbors(userImg, neighbors);


    std::vector<std::pair<cv::Vec3i, cv::Vec3i>> edges = getEdges(neighbors);

    std::vector<ComponentSubimage> subImages = getSubimages(userImg, edges);

    // for (int i = 0; i < subImages.size(); ++i) {
    //     cv::imshow("Component " + std::to_string(i), subImages[i].image);
    //     cv::waitKey(0);
    // }
    return subImages;
}

// int main() {
//     SubimageGenerator generator;

//     std::vector<ComponentSubimage> subImages = generator.generateSubimages("../component_images/hough_circuit3a.jpg");
    

//     return 0;
// }
