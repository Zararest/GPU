cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(PBQP LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_library(PBQP ./lib/PBQP.cpp
                 ./lib/Utils.cpp
                 ./lib/GPU-graph.cu
                 ./lib/GPU-solver.cu)
target_link_libraries(PBQP PRIVATE CUDA::cudart CUDA::cuda_driver)
target_include_directories(PBQP PUBLIC include/)
set_target_properties(PBQP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(Test main.cpp)
target_link_libraries(Test PRIVATE PBQP)

add_executable(Graph-generator Graph-generator.cpp ./lib/Graph-gen.cpp)
target_link_libraries(Graph-generator PRIVATE PBQP)