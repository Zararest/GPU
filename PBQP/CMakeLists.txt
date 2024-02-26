cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(PBQP LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_executable(Test main.cpp ./lib/PBQP.cpp)
target_include_directories(Test PUBLIC include/)
set_target_properties(Test PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
