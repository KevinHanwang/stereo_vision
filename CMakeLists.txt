cmake_minimum_required(VERSION 2.8.3)
project(stereoCalibration)

SET( CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(OpenCV_DIR /opt/ros/kinetic/share/OpenCV-3.3.1-dev)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

add_executable(imgSeg imgSeg.cpp)
target_link_libraries(imgSeg ${OpenCV_LIBRARIES})

add_executable(stereoDepthCal stereoDepthCal.cpp)
target_link_libraries(stereoDepthCal ${OpenCV_LIBRARIES})

add_executable(ORBExtractorTest ORBExtractorTest.cpp)
target_link_libraries(ORBExtractorTest ${OpenCV_LIBRARIES})

add_executable(test test.cpp)
target_link_libraries(test ${OpenCV_LIBRARIES})

add_executable(stereoDemo stereoDemo.cpp)
target_link_libraries(stereoDemo ${OpenCV_LIBRARIES})