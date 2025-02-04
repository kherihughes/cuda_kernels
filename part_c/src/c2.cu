#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include "include/cuda_utils.h"
#include "include/convolution_ops.h"

#define C 3
#define H 1024
#define W 1024
#define FH 3
#define FW 3
#define K 64
#define P 1
#define MILLION 1000000L
#define TILE_SIZE 16
#define BLOCK_SIZE (TILE_SIZE + FW - 1)

__global__ void convolution(float *I0, float *F, float *O) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int bx = blockIdx.x;
    int by = blockIdx.y;
    int k = blockIdx.z;

    // Use more registers for frequently accessed values
    const int x = bx * TILE_SIZE + tx;
    const int y = by * TILE_SIZE + ty;
    const int x_in = x + P;
    const int y_in = y + P;

    // Shared memory for input tile and halo elements
    __shared__ float I_shared[C][BLOCK_SIZE][BLOCK_SIZE];

    // Load input data into shared memory
    if (x < W && y < H) {
        #pragma unroll
        for (int c = 0; c < C; ++c) {
            int global_index = c * (H + 2 * P) * (W + 2 * P) + y_in * (W + 2 * P) + x_in;
            I_shared[c][ty][tx] = I0[global_index];
        }
    }
     __syncthreads();

    // Perform convolution
    if (x < W && y < H) {
        float sum = 0.0;
        #pragma unroll
        for (int c = 0; c < C; ++c) {
             #pragma unroll
            for (int i = 0; i < FH; ++i) {
                 #pragma unroll
                for (int j = 0; j < FW; ++j) {
                    int filter_index = k * C * FH * FW + c * FH * FW + (FH - 1 - i) * FW + (FW - 1 - j);

                    int shared_x = tx + j - P;
                    int shared_y = ty + i - P;
                    if(shared_x >= 0 && shared_x < BLOCK_SIZE && shared_y >=0 && shared_y < BLOCK_SIZE)
                     sum += F[filter_index] * I_shared[c][shared_y][shared_x];
                    
                }
            }
        }
        O[k * W * H + y * W + x] = sum;
    }
}

int main() {
    // Generate input tensor I and convolution filters F
    int nI = C * H * W;
    int nF = K * C * FH * FW;
    int nI0 = C * (W + 2 * P) * (H + 2 * P);
    int nO = K * W * H;

    float *I = (float*)malloc(nI * sizeof(float));
    float *F = (float*)malloc(nF * sizeof(float));
    // Initialize I and F
    for(int c=0;c<C;c++){
        for(int x=0;x<H;x++){
            for(int y=0;y<W;y++){
                I[c * H * W + x * W + y] = c * (x + y);
            }
        }
    }

    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    F[k * C * FH * FW + c * FH * FW + i * FW + j] = (c + k) * (i + j);
                }
            }
        }
    }

    // Pad input tensor I to obtain I0
    float *I0 = (float*)malloc(nI0 * sizeof(float));
    // Pad I to obtain I0
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H + 2 * P; ++x) {
            for (int y = 0; y < W + 2 * P; ++y) {
                if (x < P || x >= H + P || y < P || y >= W + P) {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = 0.0f; // Padding with zeros
                } else {
                    I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = I[c * H * W + (x - P) * W + (y - P)];
                }
            }
        }
    }

    // Allocate memory on GPU
    float *d_I0, *d_F, *d_O;
    cudaMalloc(&d_I0, sizeof(float) * nI0);
    cudaMalloc(&d_F, sizeof(float) * nF);
    cudaMalloc(&d_O, sizeof(float) * nO);

    // Copy data from host to GPU
    cudaMemcpy(d_I0, I0, sizeof(float) * nI0, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, sizeof(float) * nF, cudaMemcpyHostToDevice);

    // Define grid and block dimensions
    dim3 blockDim(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridDim((W + TILE_SIZE - 1) / TILE_SIZE, (H + TILE_SIZE - 1) / TILE_SIZE, K);

    // Launch the kernel, warm-up
    convolution<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    // Synchronize threads
    cudaDeviceSynchronize();

    struct timespec start, end;
    double total_time = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    convolution<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);
    total_time = (end.tv_sec - start.tv_sec)*1000 + (end.tv_nsec - start.tv_nsec) / (double)MILLION;

    // Copy the result back to host
    float *O_host = (float*)malloc(nO * sizeof(float));
    cudaMemcpy(O_host, d_O, sizeof(float) * nO, cudaMemcpyDeviceToHost);

    // Compute checksum
    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                checksum += O_host[k * W * H + x * H + y];
            }
        }
    }
    printf("C2_checksum: %.6e\n", checksum);
    printf("C2_execution_time: %.3f ms\n", total_time);

    // Free memory
    free(I);
    free(F);
    free(I0);
    free(O_host);
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);

    return 0;
}