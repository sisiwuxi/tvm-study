cmake_minimum_required(VERSION 3.9 FATAL_ERROR)
set(CMAKE_CUDA_COMPILER "/usr/local/cuda-9.0/bin/nvcc")
set(CMAKE_CUDA_COMPILER_VERSION "9.0")
set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "/usr/local/cuda-9.0")
set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "/usr/local/cuda-9.0")
set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "/usr/local/cuda-9.0/targets/aarch64-linux/include")
project(cu_prog CXX C CUDA)
#set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G -O3")
set(CMAKE_CUDA_FLAGS "-arch=compute_35 -g -G")
find_package(CUDA REQUIRED)
include_directories(./include)
add_subdirectory(how_many)
add_subdirectory(atomic)
add_subdirectory(cube)
add_subdirectory(add_kernel)
add_subdirectory(parallel_reduction)
add_subdirectory(transpose)
add_subdirectory(0_malloc_pitch)
add_subdirectory(1_add)
add_subdirectory(2_conv)
add_subdirectory(3_sum)
add_subdirectory(3_reduce_sum)
add_subdirectory(4_histogram)
add_subdirectory(5_thrust)
add_subdirectory(vecadd)
add_subdirectory(matadd)
add_subdirectory(cudaMallocPitch)
add_subdirectory(stream)
add_subdirectory(texture)
add_subdirectory(performance)
# target_link_libraries(cu_prog)
