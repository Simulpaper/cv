#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#include "DatasetParser.hpp"
#include "HelperStructs.hpp"

namespace fs = std::filesystem;

std::vector<DatasetComponent> DatasetParser::getDataset(cv::Ptr<cv::ORB> orb, const cv::Vec3i& bilParams, int tLower, int tUpper) {
    const std::filesystem::path datasetDir{"../component_dataset"};
    std::vector<DatasetComponent> dataset;
    int datasetSize = 0;
    int datasetBytes = 0;
    int numDescriptors = 0;
    
    for (const auto& entry : fs::directory_iterator{datasetDir}) {
        if (!entry.is_regular_file()) {
            std::cout << "Warning: file " << entry.path() << "not a regular file! Skipping it" << std::endl;
            continue;
        }

        cv::Mat datasetImg = cv::imread(entry.path().string(), cv::IMREAD_GRAYSCALE);
        cv::Mat bil = datasetImg;
        
        // Median Blur
        cv::medianBlur(bil, bil, 3);

        // Non-Local Means Denoising
        cv::fastNlMeansDenoising(bil, bil, 30, 11, 21);

        // Adaptive Thresholding
        cv::adaptiveThreshold(bil, bil, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 191, 5);

        // Median Blur Again
        cv::medianBlur(bil, bil, 3);

        // cv::imshow("Preprocessed", bil);
        // cv::waitKey(0);

        cv::Mat datasetEdge;
        cv::Canny(bil, datasetEdge, tLower, tUpper);

        std::vector<cv::KeyPoint> keypoints;
        cv::Mat descriptors;
        orb->detectAndCompute(datasetEdge, cv::noArray(), keypoints, descriptors);
        DatasetComponent dsComponent;
        dsComponent.name = entry.path().stem().string().substr(0, entry.path().stem().string().find_first_of("0123456789"));
        dsComponent.image = datasetEdge;
        dsComponent.keypoints = keypoints;
        dsComponent.descriptors = descriptors;
        dataset.push_back(dsComponent);
        datasetSize++;
        datasetBytes += descriptors.total() * descriptors.elemSize();
        numDescriptors += descriptors.rows;
    }

    return dataset;
}

// int main() {
//     DatasetParser dsParser;

//     cv::Ptr<cv::ORB> orb = cv::ORB::create();
//     cv::Vec3i bilParams(5, 25, 25);
//     int tLower = 150;
//     int tUpper = 500;

//     std::vector<DatasetComponent> dataset = dsParser.getDataset(orb, bilParams, tLower, tUpper);

//     // Print the results or process the dataset as needed
//     for (const auto& item : dataset) {
//         std::cout << "Component type: " << item.name << std::endl;
//     }

//     return 0;
// }
