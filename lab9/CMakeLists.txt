cmake_minimum_required(VERSION 3.10)
project(lab9_prelucrare_imagini_frecvente)

set(CMAKE_CXX_STANDARD 17)

# Find OpenCV package
find_package(OpenCV REQUIRED)

# Include OpenCV headers
include_directories(${OpenCV_INCLUDE_DIRS})

# Add executable for lab9
add_executable(lab9 main_lab9.cpp frequency_domain_helpers.cpp)

# Link OpenCV libraries
target_link_libraries(lab9 ${OpenCV_LIBS})
