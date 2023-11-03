#ifndef COMPONENT_CLASSIFIER_H
#define COMPONENT_CLASSIFIER_H

#include <opencv2/opencv.hpp>
#include "HelperStructs.hpp"
#include <vector>

class ComponentClassifier {
public:
    std::vector<ComponentMatch> getClassifications(cv::Ptr<cv::ORB> orb, const cv::Vec3i& bilParams, int tLower, int tUpper, cv::Mat img, const std::vector<DatasetComponent>& dataset);
};

#endif