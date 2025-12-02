cmake_minimum_required(VERSION 3.10)
project(lab8_prelucrare_imagini)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable for lab8
add_executable(lab8 main_lab8.cpp)

# Link OpenCV libraries
target_link_libraries(lab8 ${OpenCV_LIBS})
