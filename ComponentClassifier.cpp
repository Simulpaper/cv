#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#include "ComponentClassifier.hpp"
#include "DatasetParser.hpp"
#include "HelperStructs.hpp"

namespace fs = std::filesystem;

std::vector<ComponentMatch> ComponentClassifier::getClassifications(cv::Ptr<cv::ORB> orb, const cv::Vec3i& bilParams, int tLower, int tUpper, cv::Mat img, const std::vector<DatasetComponent>& dataset) {
    cv::Mat bil = img;
    // cv::bilateralFilter(datasetImg, bil, bilParams[0], bilParams[1], bilParams[2]);

    cv::Mat edgeImage;
    cv::Canny(bil, edgeImage, tLower, tUpper);

    // cv::imshow("Canny-edged image", edgeImage);
    // cv::waitKey(0);

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(edgeImage, cv::noArray(), keypoints, descriptors);

    cv::BFMatcher matcher(cv::NORM_HAMMING, false);

    std::vector<ComponentMatch> datasetMatches;

    for (const auto& item : dataset) {
        std::vector<cv::DMatch> matches;
        matcher.match(descriptors, item.descriptors, matches, cv::Mat());
        float avgDist = 0;
        for (const auto& match : matches) {
            avgDist += match.distance;
        }
        if (matches.size() == 0) {
            std::cout << "NO MATCHES FOUND WITH " << item.name << std::endl;
            continue;
        }
        avgDist /= matches.size();
        ComponentMatch m;
        m.name = item.name;
        m.avgDist = avgDist;
        m.numMatches = matches.size();
        datasetMatches.push_back(m);
    }

    std::sort(datasetMatches.begin(), datasetMatches.end(), compareComponentMatches);

    std::vector<ComponentMatch> topThreeMatches;
    std::set<std::string> haveComponents;
    for (const auto& match : datasetMatches) {
        if (haveComponents.find(match.name) == haveComponents.end()) {
            haveComponents.insert(match.name);
            topThreeMatches.push_back(match);
            std::cout << "Component match: " << match.name  << " with num matches: " << match.numMatches << " and avg dist: " << match.avgDist << std::endl;
        }
        if (haveComponents.size() == 3) {
            std::cout << std::endl;
            break;
        }
    }
    
    return topThreeMatches;
}

// int main() {
//     DatasetParser dsParser;
//     ComponentClassifier compClassifier;

//     cv::Ptr<cv::ORB> orb = cv::ORB::create();
//     cv::Vec3i bilParams(5, 25, 25);
//     int tLower = 150;
//     int tUpper = 500;

//     std::vector<DatasetComponent> dataset = dsParser.getDataset(orb, bilParams, tLower, tUpper);
//     cv::Mat img = cv::imread("../generated_components/component3.jpg");
//     std::vector<ComponentMatch> topThreeMatches = compClassifier.getClassifications(orb, bilParams, tLower, tUpper, img, dataset);

//     return 0;
// }
