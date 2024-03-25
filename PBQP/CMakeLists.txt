cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(PBQP LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_subdirectory(lib)
target_include_directories(PBQP PUBLIC include/)

add_executable(Test main.cpp)
target_link_libraries(Test PRIVATE PBQP)

add_executable(Graph-generator Graph-generator.cpp ./lib/Graph-gen.cpp)
target_link_libraries(Graph-generator PRIVATE PBQP)

add_executable(Perf-measure Perf-measurements.cpp)
target_link_libraries(Perf-measure PRIVATE PBQP)