add_library(PBQP PBQP.cpp
                 Utils.cpp
                 GPU-graph.cu
                 GPU-solver.cu)
target_link_libraries(PBQP PRIVATE CUDA::cudart CUDA::cuda_driver)
set_target_properties(PBQP PROPERTIES CUDA_SEPARABLE_COMPILATION ON)