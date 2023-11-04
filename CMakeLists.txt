cmake_minimum_required(VERSION 3.0)
PROJECT(circuit_classifier)

find_package(OpenCV REQUIRED)

set(SOURCES
    CircuitClassifier.cpp
    ComponentClassifier.cpp
    DatasetParser.cpp
    SubimageGenerator.cpp
)

set(HEADERS
    CircuitClassifier.hpp
    ComponentClassifier.hpp
    SubimageGenerator.hpp
    DatasetParser.hpp
    HelperStructs.hpp
)

add_executable(circuit_classifier ${SOURCES} ${HEADERS})

target_include_directories(circuit_classifier PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(circuit_classifier ${OpenCV_LIBS})

