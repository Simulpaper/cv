cmake_minimum_required(VERSION 3.0)
PROJECT(dataset_creator)

set(OpenCV_DIR "/mnt/c/Users/steph/Downloads/opencv/build/x64/vc16/lib")

SET(GCC_COVERAGE_COMPILE_FLAGS "-std=c++17")
add_definitions(${GCC_COVERAGE_COMPILE_FLAGS})

find_package(OpenCV REQUIRED)

set(SOURCES
    DatasetCreator.cpp
)

set(HEADERS
    DatasetCreator.hpp
)

add_executable(dataset_creator ${SOURCES} ${HEADERS})

target_include_directories(dataset_creator PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(dataset_creator ${OpenCV_LIBS})

