add_library(timer STATIC
    timer/timer.cu
    timer/timer.h
)
set_target_properties(timer PROPERTIES 
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)
target_include_directories(timer PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/timer)