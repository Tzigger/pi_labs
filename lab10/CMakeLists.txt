cmake_minimum_required(VERSION 3.10)
project(Lab10_Segmentare)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Găsim OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Adăugăm executabilul
add_executable(lab10 
    main_lab10.cpp 
    segmentation_helpers.cpp
)

# Link cu OpenCV
target_link_libraries(lab10 ${OpenCV_LIBS})

# Mesaj informativ
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")
