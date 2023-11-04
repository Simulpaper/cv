#ifndef CIRCUIT_CLASSIFIER_H
#define CIRCUIT_CLASSIFIER_H

#include "HelperStructs.hpp"
#include <vector>

class CircuitClassifier {
public:
    std::vector<std::vector<Component>> getCircuits(std::string imageName);
};

#endif