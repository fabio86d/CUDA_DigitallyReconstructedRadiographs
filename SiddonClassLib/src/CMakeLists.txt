project(SiddonGpu)

find_package(CUDA REQUIRED)

cuda_add_library(SiddonGpu SiddonLib/siddon_class.cu SiddonLib/siddon_class.cuh)
target_link_libraries(SiddonGpu cudart cudadevrt)