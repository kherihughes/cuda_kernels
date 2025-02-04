# Find CUDA and cuDNN
find_package(CUDA REQUIRED)
find_library(CUDNN_LIBRARY cudnn REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})

# Basic Convolution
add_executable(c1 src/c1.cu)
set_target_properties(c1 PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Tiled Convolution
add_executable(c2 src/c2.cu)
set_target_properties(c2 PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# cuDNN Convolution (only if cuDNN is found)
if(CUDNN_LIBRARY)
    message(STATUS "Building c3 with cuDNN: ${CUDNN_LIBRARY}")
    add_executable(c3 src/c3.cu)
    target_include_directories(c3 PRIVATE 
        ${CMAKE_SOURCE_DIR}
        ${CUDA_INCLUDE_DIRS}
    )
    target_link_libraries(c3 ${CUDNN_LIBRARY} ${CUDA_LIBRARIES})
    set_target_properties(c3 PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
    )
else()
    message(WARNING "cuDNN not found - skipping c3 build")
endif()

#set(CUDNN_LIBRARY "/usr/lib/x86_64-linux-gnu/libcudnn.so.8") # Manual cudnn path, use only if find_library fails
# Include directories
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)