cmake_minimum_required(VERSION 3.10)

# Set CUDA compiler
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)

project(HPML_Assignment3 CUDA CXX)

# Set CUDA architectures policy
cmake_policy(SET CMP0104 NEW)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 86)
endif()

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo")

# CUDA settings
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
find_package(CUDA REQUIRED)

# Set default CUDA flags for optimization
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math -Xcompiler -O3")

# Common include directories
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_SOURCE_DIR}/include
    ${CMAKE_SOURCE_DIR}/common
    ${CMAKE_SOURCE_DIR}/common/timer
    ${CUDA_INCLUDE_DIRS}
)

# Add timer library
add_library(timer STATIC
    common/timer/timer.cu
    common/timer/timer.h
)
set_target_properties(timer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_ARCHITECTURES 75
)

# Add subdirectories
add_subdirectory(part_a)
add_subdirectory(part_b)
add_subdirectory(part_c)

# Print build information
message(STATUS "CUDA_TOOLKIT_ROOT_DIR = ${CUDA_TOOLKIT_ROOT_DIR}")
message(STATUS "CUDA_INCLUDE_DIRS = ${CUDA_INCLUDE_DIRS}")
message(STATUS "CMAKE_CUDA_COMPILER = ${CMAKE_CUDA_COMPILER}")