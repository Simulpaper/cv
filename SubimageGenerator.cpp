#include <opencv2/opencv.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>
#include <tuple>
#include "SubimageGenerator.hpp"


cv::Mat SubimageGenerator::loadImage(const std::string& filename) {
    cv::Mat img = cv::imread(filename, cv::IMREAD_GRAYSCALE);
    if (img.cols > img.rows) {
        cv::resize(img, img, cv::Size(1920, 1080), cv::INTER_AREA);
    }
    else {
        cv::resize(img, img, cv::Size(1080, 1920), cv::INTER_AREA);
    }
    return img;
}

cv::Mat SubimageGenerator::applyThreshold(const cv::Mat& img, int thresholdValue = 60) {
    cv::Mat binaryImg = img.clone();
    cv::threshold(binaryImg, binaryImg, thresholdValue, 255, cv::THRESH_BINARY);
    return binaryImg;
}

cv::Mat SubimageGenerator::applyMedianBlur(const cv::Mat& img, int ksize = 23) {
    cv::Mat blurredImg = img.clone();
    cv::medianBlur(blurredImg, blurredImg, ksize);
    return blurredImg;
}

std::vector<cv::Vec3i> SubimageGenerator::getCircles(const cv::Mat& img) {
    int rows = img.rows;
    int cols = img.cols;
    cv::Mat circlesMat;
    cv::HoughCircles(img, circlesMat, cv::HOUGH_GRADIENT, 1, std::min(rows / 8, cols / 8), 500, 10, 10, 45);

    std::vector<cv::Vec3i> circles;
    if (!circlesMat.empty()) {
        circlesMat = circlesMat.reshape(3, circlesMat.rows);
        for (int i = 0; i < circlesMat.rows; i++) {
            cv::Vec3i c = circlesMat.at<cv::Vec3i>(i);
            circles.push_back(c);
        }
    }
    else {
        std::cout << "NO CIRCLES DETECTED!!" << std::endl;
        exit(EXIT_FAILURE);
    }
    return circles;
}

void SubimageGenerator::showCircles(const cv::Mat& img, const std::vector<cv::Vec3i>& circles) {
    cv::Mat circlesImg = img.clone();
    for (const cv::Vec3i& circle : circles) {
        int x = circle[0];
        int y = circle[1];
        int r = circle[2];
        cv::circle(circlesImg, cv::Point(x, y), 1, cv::Scalar(0, 100, 100), 3);
        cv::circle(circlesImg, cv::Point(x, y), r, cv::Scalar(255, 0, 255), 3);
    }
    cv::imshow("detected circles", circlesImg);
    cv::waitKey(0);
}

double SubimageGenerator::distance(const cv::Vec3i& circle1, const cv::Vec3i& circle2) {
    int x1 = circle1[0];
    int y1 = circle1[1];
    int x2 = circle2[0];
    int y2 = circle2[1];
    return std::sqrt(static_cast<double>((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1));
}

std::vector<cv::Vec3i> getNeighbors(const cv::Vec3i& circle, const std::vector<cv::Vec3i>& circles) {
    std::vector<cv::Vec3i> neighbors(4, cv::Vec3i(0, 0, 0));
    int thisX = circle[0];
    int thisY = circle[1];
    int thisR = circle[2];
    int distDifferential = 50;

    for (const cv::Vec3i& otherCircle : circles) {
        if (circle == otherCircle) {
            continue;
        }
        int otherX = otherCircle[0];
        int otherY = otherCircle[1];
        int otherR = otherCircle[2];
        int upDist = thisY - otherY;
        int downDist = otherY - thisY;
        int leftDist = thisX - otherX;
        int rightDist = otherX - thisX;
        std::vector<int> distances = {leftDist, rightDist, downDist, upDist};
        double eDist = distance(circle, otherCircle);

        for (int i = 0; i < 4; i++) {
            int dirDist = distances[i];
            if (dirDist > 0 && eDist - dirDist <= distDifferential && (neighbors[i][0] == 0 || dirDist < neighbors[i][1])) {
                neighbors[i] = otherCircle;
            }
        }
    }

    return neighbors;
}

int main() {
    ImageAndCircuitProcessor processor;

    cv::Mat userImg = processor.loadImage("component_images/hough_circuit3a.jpg");
    cv::Mat thresholdedImg = processor.applyThreshold(userImg);
    cv::Mat medBlurredImg = processor.applyMedianBlur(thresholdedImg);
    std::vector<cv::Vec3i> circles = processor.getCircles(medBlurredImg);
    processor.showCircles(userImg, circles);

    std::vector<std::vector<cv::Vec3i>> neighbors(circles.size());
    for (size_t i = 0; i < circles.size(); i++) {
        neighbors[i] = processor.getNeighbors(circles[i], circles);
    }

    // Add the remaining logic for edges, subimages, and any other processing

    return 0;
}