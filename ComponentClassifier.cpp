#include <opencv2/opencv.hpp>
#include <opencv2/xfeatures2d.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#include "ComponentClassifier.hpp"
#include "DatasetParser.hpp"
#include "HelperStructs.hpp"

namespace fs = std::filesystem;

std::vector<ComponentMatch> ComponentClassifier::getClassifications(cv::Ptr<cv::ORB> orb, int tLower, int tUpper, cv::Mat img, const std::vector<DatasetComponent>& dataset) {
    cv::Mat bil = img;
    
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

    cv::Mat edgeImage;
    cv::Canny(bil, edgeImage, tLower, tUpper);

    // cv::imshow("Canny-edged image", edgeImage);
    // cv::waitKey(0); 

    int rows = edgeImage.rows;
    int cols = edgeImage.cols;

    cv::Mat circles;
    cv::HoughCircles(edgeImage, circles, cv::HOUGH_GRADIENT, 1, std::max(rows, cols),
                     300, 30, (std::min(rows, cols) / 4), (std::min(rows, cols)) / 2 * 1.2);

    std::set<std::string> toCompare;
    // if is a component with a circle
    if (!circles.empty()) {
        toCompare = {"voltagesourceu", "voltagesourced", "voltagesourcel", "voltagesourcer", "currentsourceu", "currentsourced", "currentsourcel", "currentsourcer", "lightbulb", "resistor"};
        std::cout << "Circle in component detected!" << std::endl;
    // is a component with no circle
    } else {
        std::vector<cv::Vec4i> linesP;
        cv::HoughLinesP(255 - bil, linesP, 2, CV_PI / 180, 50, 50, 10);
        int lowX = cols;
        int highX = 0;
        int lowY = rows;
        int highY = 0;

        if (!linesP.empty()) {
            for (int i = 0; i < linesP.size(); i++) {
                cv::Vec4i l = linesP[i];
                lowX = std::min(lowX, std::min(l[0], l[2]));
                lowY = std::min(lowY, std::min(l[1], l[3]));
                highX = std::max(highX, std::max(l[0], l[2]));
                highY = std::max(highY, std::max(l[1], l[3]));
            }

            // img is vertical && horizontal diff is greater than width/3 OR img is horizontal && vertical diff is greater than height/3
            if ((rows > cols && highX - lowX > cols / 3) ||
                (cols > rows && highY - lowY > rows / 3)) {
                toCompare = {"resistor", "diodeu", "dioded", "diodel", "dioder", "switch"};
            } else {
                // is a wire
                ComponentMatch match;
                match.name = "wire";
                match.avgDist = 1;
                match.numMatches = 1;
                std::vector<ComponentMatch> matches{match};
                std::cout << "Component match: " << match.name  << " with num matches: " << match.numMatches << " and avg dist: " << match.avgDist << std::endl << std::endl;
                return matches;
            }
        // if couldn't detect any lines, assume wire
        } else {
            ComponentMatch match;
            match.name = "wire";
            match.avgDist = 1;
            match.numMatches = 1;
            std::vector<ComponentMatch> matches{match};
            std::cout << "Component match: " << match.name  << " with num matches: " << match.numMatches << " and avg dist: " << match.avgDist << std::endl << std::endl;
            return matches;
        }
    }

    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
    orb->detectAndCompute(edgeImage, cv::noArray(), keypoints, descriptors);

    cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> brief = cv::xfeatures2d::BriefDescriptorExtractor::create();
    brief->compute(edgeImage, keypoints, descriptors);


    cv::BFMatcher matcher(cv::NORM_HAMMING, false);

    std::vector<ComponentMatch> datasetMatches;

    for (const auto& item : dataset) {
        if (toCompare.count(item.name) == 0) {
            continue;
        }
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

    // if no dataset matches or avg distance of most matching component is >= 40, run against all component types
    // if (datasetMatches.empty() || datasetMatches[0].avgDist >= 45) {
    //     for (const auto& item : dataset) {

    //         std::vector<cv::DMatch> matches;
    //         matcher.match(descriptors, item.descriptors, matches, cv::Mat());
    //         float avgDist = 0;
    //         for (const auto& match : matches) {
    //             avgDist += match.distance;
    //         }
    //         if (matches.size() == 0) {
    //             std::cout << "NO MATCHES FOUND WITH " << item.name << std::endl;
    //             continue;
    //         }
    //         avgDist /= matches.size();
    //         ComponentMatch m;
    //         m.name = item.name;
    //         m.avgDist = avgDist;
    //         m.numMatches = matches.size();
    //         datasetMatches.push_back(m);
    //     }
    // }

    std::sort(datasetMatches.begin(), datasetMatches.end(), compareComponentMatches);

    // if still no dataset matches, default wire
    if (datasetMatches.empty()) {
        ComponentMatch match;
        match.name = "wire";
        match.avgDist = 1;
        match.numMatches = 1;
        datasetMatches.push_back(match);
    }

    std::vector<ComponentMatch> topThreeMatches;
    std::set<std::string> haveComponents;
    for (const auto& match : datasetMatches) {
        if (haveComponents.find(match.name) == haveComponents.end()) {
            haveComponents.insert(match.name);
            topThreeMatches.push_back(match);
            std::cout << "Component match: " << match.name  << " with num matches: " << match.numMatches << " and avg dist: " << match.avgDist << std::endl;
        }
        if (haveComponents.size() == 3) {
            break;
        }
    }
    std::cout << std::endl;
    
    return topThreeMatches;
}

// int main() {
//     DatasetParser dsParser;
//     ComponentClassifier compClassifier;

//     cv::Ptr<cv::ORB> orb = cv::ORB::create();
//     int tLower = 150;
//     int tUpper = 500;

//     std::vector<DatasetComponent> dataset = dsParser.getDataset(orb, tLower, tUpper);
//     cv::Mat img = cv::imread("../generated_components/component3.jpg");
//     std::vector<ComponentMatch> topThreeMatches = compClassifier.getClassifications(orb, tLower, tUpper, img, dataset);

//     return 0;
// }
