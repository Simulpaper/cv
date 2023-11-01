#ifndef DATASET_PARSER_H
#define DATASET_PARSER_H

#include <opencv2/opencv.hpp>
#include "HelperStructs.hpp"
#include <vector>

class DatasetParser {
public:
    std::vector<DatasetComponent> getDataset(cv::Ptr<cv::ORB> orb, const cv::Vec3i& bilParams, int tLower, int tUpper);
};

#endif