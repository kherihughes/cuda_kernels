# CPU Vector Addition
add_executable(q1 src/q1.cpp)
set_target_properties(q1 PROPERTIES CXX_STANDARD 14)

# GPU Vector Addition without Unified Memory
add_executable(q2 src/q2.cu)
set_target_properties(q2 PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# GPU Vector Addition with Unified Memory
add_executable(q3 src/q3.cu)
set_target_properties(q3 PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)