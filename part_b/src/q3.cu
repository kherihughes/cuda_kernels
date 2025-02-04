#include <iostream>
#include <cuda_runtime.h>
#include "include/cuda_utils.h"

// Unique kernel for q3
__global__ void AddVectorsQ3(const float *A, const float *B, float *C, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int totalThreads = blockDim.x * gridDim.x;

    // Ensure each thread accesses only valid indices
    for (int i = tid; i < N; i += totalThreads) {
        C[i] = A[i] + B[i];
    }
}

void executeVectorAddQ3(int N, int numBlocks, int threadsPerBlock) {
    // Allocate unified memory
    float *a, *b, *c;
    cudaMallocManaged(&a, N * sizeof(float));
    cudaMallocManaged(&b, N * sizeof(float));
    cudaMallocManaged(&c, N * sizeof(float));

    // Initialize arrays
    for (int i = 0; i < N; i++) {
        a[i] = static_cast<float>(i);
        b[i] = static_cast<float>(i);
    }

    // Warmup kernel execution to avoid cold start overhead
    AddVectorsQ3<<<numBlocks, threadsPerBlock>>>(a, b, c, N);
    cudaDeviceSynchronize();

    // Timing setup using CUDA events
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Start recording
    cudaEventRecord(start, 0);

    // Main kernel execution
    AddVectorsQ3<<<numBlocks, threadsPerBlock>>>(a, b, c, N);

    // Stop recording
    cudaEventRecord(stop, 0);
    cudaDeviceSynchronize();

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Display time taken
    std::cout << "Time taken for N = " << N << " elements ("
              << numBlocks << " blocks, " << threadsPerBlock << " threads per block): "
              << milliseconds << " ms" << std::endl;

    // Clean up
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}

int main(int argc, char **argv) {
    int K = (argc > 1) ? atoi(argv[1]) : 1;
    int N = K * 1000000;

    // Scenario 1: One block with 1 thread
    executeVectorAddQ3(N, 1, 1);

    // Scenario 2: One block with 256 threads
    executeVectorAddQ3(N, 1, 256);

    // Scenario 3: Multiple blocks with 256 threads per block
    int numBlocks = (N + 255) / 256;
    executeVectorAddQ3(N, numBlocks, 256);

    return 0;
}
