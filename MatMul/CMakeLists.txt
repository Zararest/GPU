cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Example LANGUAGES CXX CUDA)

add_executable(MatMul 
               main.cu
               Utils.cpp
               Mat-mul-simple.cu
               Mat-mul-tiled.cu
               Mat-mul-tiled-transpose.cu)
          
add_executable(Example-shared
               cuda-guide/with-shared.cu)       
               
add_executable(Example-simple
               cuda-guide/without-shared.cu)

target_compile_features(MatMul PUBLIC cxx_std_11)

set_target_properties(MatMul
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)