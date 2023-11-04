#pragma once

#include <opencv2/opencv.hpp>
#include <vector>


struct ComponentSubimage {
    cv::Vec3i node1;
    cv::Vec3i node2;
    cv::Mat image;
};

struct Vec3iCompare {
        bool operator() (const cv::Vec3i& circle1, const cv::Vec3i& circle2) const {
            return circle1[0] + circle1[1] < circle2[0] + circle2[1];
        }
};

struct DatasetComponent {
    std::string name;
    cv::Mat image;
    std::vector<cv::KeyPoint> keypoints;
    cv::Mat descriptors;
};

struct ComponentMatch {
    std::string name;
    int numMatches;
    float avgDist;
};

inline bool compareComponentMatches(const ComponentMatch& a, const ComponentMatch& b) {
    if (a.numMatches > b.numMatches) {
        return true;
    } else if (a.numMatches == b.numMatches) {
        return a.avgDist < b.avgDist;
    }
    return false;
}

struct Component {
    std::pair<int, int> firstNode;
    std::pair<int, int> secondNode;
    std::string type;
};

struct CircuitClassification {
    std::vector<Component> edges;
    int score;
};

struct ClassifiedEdge {
    std::pair<int, int> firstNode;
    std::pair<int, int> secondNode;
    std::vector<ComponentMatch> classifications;
};

inline bool compareCircuits(const CircuitClassification& a, const CircuitClassification& b) {
    return a.score < b.score;
}