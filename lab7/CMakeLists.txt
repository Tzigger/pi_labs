cmake_minimum_required(VERSION 3.10)
project(lab7_prelucrare_imagini)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable for lab7
add_executable(lab7 main.cpp)

# Link OpenCV libraries
target_link_libraries(lab7 ${OpenCV_LIBS})

# Optional: Add executable for saving files
add_executable(save_files main_save_files.cpp)
target_link_libraries(save_files ${OpenCV_LIBS})
