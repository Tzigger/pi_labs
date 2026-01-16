cmake_minimum_required(VERSION 3.10)
project(Lab13_ColorSpaces)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find OpenCV
find_package(OpenCV REQUIRED)

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable
add_executable(lab13 
    main_lab13.cpp
    color_spaces.cpp
)

# Link libraries
target_link_libraries(lab13 ${OpenCV_LIBS})

# Print OpenCV version
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include dirs: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
