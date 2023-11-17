#include <opencv2/opencv.hpp>
#include <vector>
#include <string>
#include <iostream>
#include <filesystem>

#include "CircuitClassifier.hpp"
#include "ComponentClassifier.hpp"
#include "SubimageGenerator.hpp"
#include "DatasetParser.hpp"
#include "HelperStructs.hpp"

std::vector<CircuitClassification> generateCircuits(const std::vector<ClassifiedEdge>& edges) {
    size_t numComponents = edges.size();
    std::vector<std::vector<ComponentMatch>> possibleTypes;
    for (const auto& edge : edges) {
        possibleTypes.push_back(edge.classifications);
    }
    // Vector to store all combinations
    std::vector<CircuitClassification> combinations;

    std::vector<std::vector<ComponentMatch>::iterator> it(numComponents); // Vector of iterators

    // Initialize iterators
    for (int i = 0; i < numComponents; ++i) {
        it[i] = possibleTypes[i].begin();
    }
    int count = 0;
    // Loop to generate combinations
    while (it[0] != possibleTypes[0].end()) {
        // Create a new combination
        CircuitClassification combination;
        std::vector<Component> newEdges;
        float score = 0;
        for (size_t i = 0; i < numComponents; ++i) {
            ComponentMatch cMatch = *it[i];
            Component c;
            c.firstNode = edges[i].firstNode;
            c.secondNode = edges[i].secondNode;
            c.type = cMatch.name;
            score += cMatch.avgDist;
            newEdges.push_back(c);
        }
        combination.edges = newEdges;
        combination.score = score;

        combinations.push_back(combination);
        count++;

        // Increment the last iterator
        ++it[numComponents - 1];

        // Check and update other iterators as needed
        for (int i = numComponents - 1; i > 0 && it[i] == possibleTypes[i].end(); --i) {
            it[i] = possibleTypes[i].begin();
            ++it[i - 1];
        }
    }
    std::cout << "number of combos: " << count << std::endl;
    std::sort(combinations.begin(), combinations.end(), compareCircuits);
    return combinations;
}

std::vector<std::vector<Component>> CircuitClassifier::getCircuits(std::string imageName) {
    SubimageGenerator siGen;
    DatasetParser dsParser;
    ComponentClassifier compClassifier;
    std::vector<ComponentSubimage> subimages = siGen.generateSubimages(imageName);

    cv::Ptr<cv::ORB> orb = cv::ORB::create();
    int tLower = 100;
    int tUpper = 300;

    std::vector<DatasetComponent> dataset = dsParser.getDataset(orb, tLower, tUpper);
    if (dataset.size() == 0) {
        std::cout << "ERROR: dataset is not found/corrupted!" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::vector<ClassifiedEdge> classifiedEdges;

    for (const auto& subimage : subimages) {
        ClassifiedEdge edge;
        // make the nodes just a pair of x, y coords instead of a Vec3i of x, y, r
        edge.firstNode = std::pair<int, int>(subimage.node1[0], subimage.node1[1]);
        edge.secondNode = std::pair<int, int>(subimage.node2[0], subimage.node2[1]);
        edge.classifications = compClassifier.getClassifications(orb, tLower, tUpper, subimage.image, dataset);

        classifiedEdges.push_back(edge);
    }

    std::vector<CircuitClassification> possibleCircuits = generateCircuits(classifiedEdges);
    std::vector<std::vector<Component>> bestCircuits;
    int numBest = std::min(5, (int)possibleCircuits.size());
    for (int i = 0; i < numBest; i++) {
        bestCircuits.push_back(possibleCircuits[i].edges);
    }
    return bestCircuits;
}

int main() {
    CircuitClassifier cClassifier;

    std::vector<std::vector<Component>> bestCircuits = cClassifier.getCircuits("../component_images/demo.jpg");
    for (const auto& circuit : bestCircuits) {
        std::cout << "Circuit edges: [" << std::endl;
        for (const auto& component : circuit) {
            std::cout << "{" << "(" << std::to_string(component.firstNode.first) << ", " << std::to_string(component.firstNode.second) << "), ";
            std::cout << "(" << std::to_string(component.secondNode.first) << ", " << std::to_string(component.secondNode.second) << "), ";
            std::cout << component.type << "}" << std::endl;
        }
        std::cout << "]" << std::endl << std::endl;
    }
    return 0;
}