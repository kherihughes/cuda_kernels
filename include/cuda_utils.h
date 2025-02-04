#pragma once

#include <stdio.h>
#include <cuda_runtime.h>

// Error checking macro for CUDA calls
#define CHECK_CUDA_ERROR(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error in %s at line %d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Memory management utilities
namespace cuda_utils {
    template<typename T>
    void allocateDeviceMemory(T** ptr, size_t size) {
        CHECK_CUDA_ERROR(cudaMalloc((void**)ptr, size * sizeof(T)));
    }

    template<typename T>
    void copyToDevice(T* dst, const T* src, size_t size) {
        CHECK_CUDA_ERROR(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyHostToDevice));
    }

    template<typename T>
    void copyToHost(T* dst, const T* src, size_t size) {
        CHECK_CUDA_ERROR(cudaMemcpy(dst, src, size * sizeof(T), cudaMemcpyDeviceToHost));
    }

    template<typename T>
    void freeDeviceMemory(T* ptr) {
        if (ptr) {
            CHECK_CUDA_ERROR(cudaFree(ptr));
        }
    }

    // Timer class for kernel profiling
    class CudaTimer {
    private:
        cudaEvent_t start_, stop_;
        
    public:
        CudaTimer() {
            CHECK_CUDA_ERROR(cudaEventCreate(&start_));
            CHECK_CUDA_ERROR(cudaEventCreate(&stop_));
        }

        ~CudaTimer() {
            cudaEventDestroy(start_);
            cudaEventDestroy(stop_);
        }

        void start() {
            CHECK_CUDA_ERROR(cudaEventRecord(start_));
        }

        float stop() {
            float milliseconds = 0;
            CHECK_CUDA_ERROR(cudaEventRecord(stop_));
            CHECK_CUDA_ERROR(cudaEventSynchronize(stop_));
            CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start_, stop_));
            return milliseconds;
        }
    };
}