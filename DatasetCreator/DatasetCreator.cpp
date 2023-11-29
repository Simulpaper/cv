#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#include "DatasetCreator.hpp"

void DatasetCreator::createDataset(cv::Ptr<cv::ORB> orb, int tLower, int tUpper) {
    const std::filesystem::path datasetDir{"../../component_dataset"};

    cv::FileStorage file("../../dataset.yml", cv::FileStorage::WRITE);

    for (const auto& entry : std::filesystem::directory_iterator{datasetDir}) {
        if (!entry.is_regular_file()) {
            std::cout << "Warning: file " << entry.path() << "not a regular file! Skipping it" << std::endl;
            continue;
        }

        cv::Mat datasetImg = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        cv::Mat bil = datasetImg;
        
        // Median Blur
        cv::medianBlur(bil, bil, 3);

        // Non-Local Means Denoising
        cv::fastNlMeansDenoising(bil, bil, 30, 7, 11);

        // Adaptive Thresholding
        cv::adaptiveThreshold(bil, bil, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 31, 5);

        // Median Blur Again
        cv::medianBlur(bil, bil, 5);

        // cv::imshow("Preprocessed", bil);
        // cv::waitKey(0);

        cv::Mat datasetEdge;
        cv::Canny(bil, datasetEdge, tLower, tUpper);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(datasetEdge, cv::noArray(), keypoints, descriptors);
        cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();
        brief->compute(edgeImage, keypoints, descriptors);

        std::cout << entry.path().stem().string() << std::endl;

        file << entry.path().stem().string() << descriptors;
    }
    file.release();
}

int main() {
    DatasetCreator dsCreator;

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    int tLower = 150;
    int tUpper = 500;

    dsCreator.createDataset(orb, tLower, tUpper);

    return 0;
}
