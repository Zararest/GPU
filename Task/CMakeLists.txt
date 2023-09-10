cmake_minimum_required(VERSION 3.8 FATAL_ERROR)

project(Example LANGUAGES CXX CUDA)

add_executable(VecAdd VecAdd.cu)

target_compile_features(VecAdd PUBLIC cxx_std_11)

set_target_properties(VecAdd
                      PROPERTIES CUDA_SEPARABLE_COMPILATION ON)