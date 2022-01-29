# Install script for directory: /home/sisi/D/git/sisiwuxi/cuda

# Set the install prefix
if(NOT DEFINED CMAKE_INSTALL_PREFIX)
  set(CMAKE_INSTALL_PREFIX "/usr/local")
endif()
string(REGEX REPLACE "/$" "" CMAKE_INSTALL_PREFIX "${CMAKE_INSTALL_PREFIX}")

# Set the install configuration name.
if(NOT DEFINED CMAKE_INSTALL_CONFIG_NAME)
  if(BUILD_TYPE)
    string(REGEX REPLACE "^[^A-Za-z0-9_]+" ""
           CMAKE_INSTALL_CONFIG_NAME "${BUILD_TYPE}")
  else()
    set(CMAKE_INSTALL_CONFIG_NAME "")
  endif()
  message(STATUS "Install configuration: \"${CMAKE_INSTALL_CONFIG_NAME}\"")
endif()

# Set the component getting installed.
if(NOT CMAKE_INSTALL_COMPONENT)
  if(COMPONENT)
    message(STATUS "Install component: \"${COMPONENT}\"")
    set(CMAKE_INSTALL_COMPONENT "${COMPONENT}")
  else()
    set(CMAKE_INSTALL_COMPONENT)
  endif()
endif()

# Install shared libraries without execute permission?
if(NOT DEFINED CMAKE_INSTALL_SO_NO_EXE)
  set(CMAKE_INSTALL_SO_NO_EXE "1")
endif()

# Is this installation the result of a crosscompile?
if(NOT DEFINED CMAKE_CROSSCOMPILING)
  set(CMAKE_CROSSCOMPILING "FALSE")
endif()

# Set default install directory permissions.
if(NOT DEFINED CMAKE_OBJDUMP)
  set(CMAKE_OBJDUMP "/usr/bin/objdump")
endif()

if(NOT CMAKE_INSTALL_LOCAL_ONLY)
  # Include the install script for each subdirectory.
  include("/home/sisi/D/git/sisiwuxi/cuda/build/0_hello_world/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/1_check_dimension/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/2_grid_block/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/3_sum_arrays/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/4_sum_arrays_timer/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/5_thread_index/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/6_sum_matrix/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/7_device_information/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/8_divergence/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/9_sum_matrix2D/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/10_reduceInteger/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/11_simple_sum_matrix2D/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/12_reduce_unrolling/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/14_global_variable/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/15_pine_memory/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/16_zero_copy_memory/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/17_UVA/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/18_sum_array_offset/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/19_AoS/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/20_SoA/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/21_sum_array_offset_unrolling/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/22_transform_matrix2D/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/23_sum_array_uniform_memory/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/24_shared_memory_read_data/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/25_reduce_integer_shared_memory/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/26_transform_shared_memory/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/27_stencil_1d_constant_read_only/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/28_shfl_test/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/29_reduce_shfl/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/30_stream/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/32_stream_resource/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/33_stream_block/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/34_stream_dependence/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/35_multi_add_depth/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/36_multi_add_breadth/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/37_asyncAPI/cmake_install.cmake")
  include("/home/sisi/D/git/sisiwuxi/cuda/build/38_stream_call_back/cmake_install.cmake")

endif()

if(CMAKE_INSTALL_COMPONENT)
  set(CMAKE_INSTALL_MANIFEST "install_manifest_${CMAKE_INSTALL_COMPONENT}.txt")
else()
  set(CMAKE_INSTALL_MANIFEST "install_manifest.txt")
endif()

string(REPLACE ";" "\n" CMAKE_INSTALL_MANIFEST_CONTENT
       "${CMAKE_INSTALL_MANIFEST_FILES}")
file(WRITE "/home/sisi/D/git/sisiwuxi/cuda/build/${CMAKE_INSTALL_MANIFEST}"
     "${CMAKE_INSTALL_MANIFEST_CONTENT}")
