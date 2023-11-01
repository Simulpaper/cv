cmake_minimum_required(VERSION 3.0)
PROJECT(SubimageGenerator)

find_package(OpenCV REQUIRED)

set(SOURCES
    DatasetParser.cpp

)

set(HEADERS
    DatasetParser.hpp
    HelperStructs.hpp
)

add_executable(subimage_generator ${SOURCES} ${HEADERS})

target_include_directories(subimage_generator PRIVATE ${CMAKE_SOURCE_DIR})

target_link_libraries(subimage_generator ${OpenCV_LIBS})

