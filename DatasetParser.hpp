#ifndef DATASET_PARSER_H
#define DATASET_PARSER_H

#include <opencv2/opencv.hpp>
#include "HelperStructs.hpp"
#include <vector>

class DatasetParser {
public:
    std::vector<DatasetComponent> getDataset(cv::Ptr<cv::ORB> orb, int tLower, int tUpper);
    std::vector<DatasetComponent> getDatasetFromFile(std::string filename);
};

#endif