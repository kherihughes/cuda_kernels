#include <iostream>
#include <cuda_runtime.h>
#include "include/cuda_utils.h"

__global__ void AddVectorsQ2(const float *A, const float *B, float *C, int N) {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    int numTotalThreads = blockDim.x * gridDim.x;

    for (int i = tid; i < N; i += numTotalThreads) {
        C[i] = A[i] + B[i];
    }
}

void executeVectorAddQ2(int N, int numBlocks, int threadsPerBlock) {
    float *h_a = new float[N];
    float *h_b = new float[N];
    float *h_c = new float[N];

    for (int i = 0; i < N; i++) {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(i);
    }

    float *d_a, *d_b, *d_c;
    CHECK_CUDA_ERROR(cudaMalloc(&d_a, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_b, N * sizeof(float)));
    CHECK_CUDA_ERROR(cudaMalloc(&d_c, N * sizeof(float)));

    CHECK_CUDA_ERROR(cudaMemcpy(d_a, h_a, N * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERROR(cudaMemcpy(d_b, h_b, N * sizeof(float), cudaMemcpyHostToDevice));

    AddVectorsQ2<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    cudaEvent_t start, stop;
    CHECK_CUDA_ERROR(cudaEventCreate(&start));
    CHECK_CUDA_ERROR(cudaEventCreate(&stop));

    CHECK_CUDA_ERROR(cudaEventRecord(start, 0));
    AddVectorsQ2<<<numBlocks, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CHECK_CUDA_ERROR(cudaEventRecord(stop, 0));
    CHECK_CUDA_ERROR(cudaDeviceSynchronize());

    float milliseconds = 0;
    CHECK_CUDA_ERROR(cudaEventElapsedTime(&milliseconds, start, stop));

    CHECK_CUDA_ERROR(cudaMemcpy(h_c, d_c, N * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Time taken for N = " << N << " elements ("
              << numBlocks << " blocks, " << threadsPerBlock << " threads per block): "
              << milliseconds << " ms" << std::endl;

    CHECK_CUDA_ERROR(cudaEventDestroy(start));
    CHECK_CUDA_ERROR(cudaEventDestroy(stop));
    CHECK_CUDA_ERROR(cudaFree(d_a));
    CHECK_CUDA_ERROR(cudaFree(d_b));
    CHECK_CUDA_ERROR(cudaFree(d_c));
    delete[] h_a;
    delete[] h_b;
    delete[] h_c;
}

int main(int argc, char **argv) {
    int K = (argc > 1) ? atoi(argv[1]) : 1;
    int N = K * 1000000;

    executeVectorAddQ2(N, 1, 1);
    
    executeVectorAddQ2(N, 1, 256);

    int numBlocks = (N + 255) / 256;
    executeVectorAddQ2(N, numBlocks, 256);

    return 0;
}
