cmake_minimum_required(VERSION 3.10)
project(lab11)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Găsește OpenCV
find_package(OpenCV REQUIRED)

# Afișează versiunea OpenCV
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV libraries: ${OpenCV_LIBS}")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})

# Adaugă executabilul
add_executable(lab11 
    main_lab11.cpp
    region_growing.cpp
)

# Link cu OpenCV
target_link_libraries(lab11 ${OpenCV_LIBS})

# Setări pentru filesystem (C++17)
if(APPLE)
    target_compile_options(lab11 PRIVATE -std=c++17)
endif()
