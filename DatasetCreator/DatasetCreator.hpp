#ifndef DATASET_CREATOR_H
#define DATASET_CREATOR_H

#include <opencv2/opencv.hpp>

class DatasetCreator {
public:
    void createDataset(cv::Ptr<cv::ORB> orb, int tLower, int tUpper);
};

#endif