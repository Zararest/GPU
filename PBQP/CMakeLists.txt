cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_COMPILER /usr/local/cuda-11/bin/nvcc)

project(PBQP LANGUAGES CXX CUDA)

set(CUDAToolkit_ROOT "/usr/local/cuda")
find_package(CUDA REQUIRED)
include_directories("/usr/local/cuda/include")

add_subdirectory(lib)
target_include_directories(PBQP PUBLIC include/)

add_executable(Graph-print Graph-printer.cpp)
target_link_libraries(Graph-print PRIVATE PBQP)

add_executable(Graph-generator Graph-generator.cpp ./lib/Graph-gen.cpp)
target_link_libraries(Graph-generator PRIVATE PBQP)

add_executable(Perf-measure Perf-measurements.cpp)
target_link_libraries(Perf-measure PRIVATE PBQP)