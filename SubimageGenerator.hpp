#ifndef SUBIMAGE_GENERATOR_H
#define SUBIMAGE_GENERATOR_H

#include <opencv2/opencv.hpp>
#include "HelperStructs.hpp"
#include <vector>

class SubimageGenerator {
public:
    cv::Mat getSubimages(const cv::Mat& img, const std::set<std::pair<cv::Vec3i, cv::Vec3i>>& edges);
    cv::Mat loadImage(const std::string& filename);
    cv::Mat applyThreshold(const cv::Mat& img, double thresholdValue = 60);
    cv::Mat applyMedianBlur(const cv::Mat& img, int ksize = 23);
    std::vector<cv::Vec3i> getCircles(const cv::Mat& img);
    void showCircles(const cv::Mat& img, const std::vector<cv::Vec3i>& circles);
    double distance(const cv::Vec3i& circle1, const cv::Vec3i& circle2);
    std::vector<cv::Vec3i> getNeighbors(const cv::Vec3i& circle, const std::vector<cv::Vec3i>& circles);
    void showNeighbors(const cv::Mat& img, const std::map<cv::Vec3i, std::vector<cv::Vec3i>, Vec3iCompare>& neighbors);
    std::vector<std::pair<cv::Vec3i, cv::Vec3i>> getEdges(const std::map<cv::Vec3i, std::vector<cv::Vec3i>, Vec3iCompare>& neighbors);
    std::vector<ComponentSubimage> getSubimages(const cv::Mat& userImg, const std::vector<std::pair<cv::Vec3i, cv::Vec3i>>& edges);
    std::vector<ComponentSubimage> generateSubimages(const std::string& filename);
};

#endif