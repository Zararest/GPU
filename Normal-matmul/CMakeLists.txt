cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Example LANGUAGES CXX CUDA)


find_package(CUDAToolkit REQUIRED)
include_directories(${CUDAToolkit_INCLUDE_DIRS})

add_executable(MatMul 
               main.cu
               Utils.cpp)
target_compile_features(MatMul PUBLIC cxx_std_11)
target_include_directories(MatMul PUBLIC include/)
set_target_properties(MatMul PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(BFS 
               BFS.cu
               Utils.cpp)
target_compile_features(BFS PUBLIC cxx_std_11)
target_include_directories(BFS PUBLIC include/)
set_target_properties(BFS PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(Errors I-love-cuda.cu)
set_target_properties(Errors PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    