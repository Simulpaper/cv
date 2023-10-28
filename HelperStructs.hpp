#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


struct ComponentSubimage {
    cv::Vec3i node1;
    cv::Vec3i node2;
    cv::Mat image;
};

struct Vec3iCompare {
        bool operator() (const cv::Vec3i& circle1, const cv::Vec3i& circle2) const {
            return circle1[0] + circle1[1] < circle2[0] + circle2[1];
        }
};

