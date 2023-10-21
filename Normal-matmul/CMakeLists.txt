cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Example LANGUAGES CXX CUDA)

add_executable(MatMul 
               main.cu
               Utils)
          
target_compile_features(MatMul PUBLIC cxx_std_11)
target_include_directories(MatMul PUBLIC include/)

set_target_properties(MatMul
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)