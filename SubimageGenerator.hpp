#ifndef SUBIMAGE_GENERATOR_H
#define SUBIMAGE_GENERATOR_H

#include <opencv2/opencv.hpp>
#include <vector>
#include <set>

class SubimageGenerator {
public:
    cv::Mat getSubimages(const cv::Mat& img, const std::set<std::pair<cv::Vec3i, cv::Vec3i>>& edges);

private:
    cv::Mat loadImage(const std::string& filename);
    cv::Mat applyThreshold(const cv::Mat& img, int thresholdValue = 60);
    cv::Mat applyMedianBlur(const cv::Mat& img, int ksize = 23);
    std::vector<cv::Vec3i> getCircles(const cv::Mat& img);
    void showCircles(const cv::Mat& img, const std::vector<cv::Vec3i>& circles);
    double distance(const cv::Vec3i& circle1, const cv::Vec3i& circle2);
    std::vector<cv::Vec3i> getNeighbors(const cv::Vec3i& circle, const std::vector<cv::Vec3i>& circles);
    std::set<std::pair<cv::Vec3i, cv::Vec3i>> getEdges(const std::vector<cv::Vec3i>& circles, const std::vector<std::vector<cv::Vec3i>>& neighbors);
    void showSubimages(const std::vector<cv::Mat>& subImages);
};

#endif