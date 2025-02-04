# Include local headers
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Matrix Multiplication Executables
add_executable(matmult00
    src/matmult.cu
    src/matmultKernel00.cu
)
target_link_libraries(matmult00 timer)
set_target_properties(matmult00 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

add_executable(matmult01
    src/matmult.cu
    src/matmultKernel01.cu
)
target_link_libraries(matmult01 timer)
set_target_properties(matmult01 PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES 75
)

# Vector Addition Executables
add_executable(vecadd00
    src/vecadd.cu
    src/vecaddKernel00.cu
)
target_link_libraries(vecadd00 timer)

add_executable(vecadd01
    src/vecadd.cu
    src/vecaddKernel01.cu
)
target_link_libraries(vecadd01 timer)

# Set default properties for all targets
foreach(target matmult00 matmult01 vecadd00 vecadd01)
    set_target_properties(${target} PROPERTIES 
        CUDA_SEPARABLE_COMPILATION ON
        CUDA_ARCHITECTURES 75
    )
    target_include_directories(${target} PRIVATE
        ${CMAKE_SOURCE_DIR}
        ${CMAKE_SOURCE_DIR}/common
        ${CMAKE_SOURCE_DIR}/common/timer
    )
endforeach()

# Dependencies for matrix multiplication targets
add_dependencies(matmult00 timer)
add_dependencies(matmult01 timer)