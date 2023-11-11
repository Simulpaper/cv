#ifndef SUBIMAGE_GENERATOR_H
#define SUBIMAGE_GENERATOR_H

#include <opencv2/opencv.hpp>
#include "HelperStructs.hpp"
#include <vector>

class SubimageGenerator {
public:
    std::vector<ComponentSubimage> generateSubimages(const std::string& filename);
};

#endif