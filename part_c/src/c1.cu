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

__global__ void basic_convolution(double *I0, double *F, double *O) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (x < W && y < H) {
        double sum = 0.0;
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < FH; ++i) {
                for (int j = 0; j < FW; ++j) {
                    int I0_x = x + i;
                    int I0_y = y + j;
                    int filter_idx = k * C * FH * FW + c * FH * FW + (FW - 1 - i) * FW + (FH - 1 - j);
                    int input_idx = c * (W + 2 * P) * (H + 2 * P) + I0_x * (H + 2 * P) + I0_y;
                    sum += F[filter_idx] * I0[input_idx];
                }
            }
        }
        O[k * W * H + x * H + y] = sum;
    }
}

int main() {
    int nI = C * H * W;
    int nF = K * C * FH * FW;
    int nI0 = C * (W + 2 * P) * (H + 2 * P);
    int nO = K * W * H;

    double *I = (double*)malloc(nI * sizeof(double));
    double *F = (double*)malloc(nF * sizeof(double));

    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H; ++x) {
            for (int y = 0; y < W; ++y) {
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

    double *I0 = (double*)malloc(nI0 * sizeof(double));
    for (int c = 0; c < C; ++c) {
        for (int x = 0; x < H + 2 * P; ++x) {
            for (int y = 0; y < W + 2 * P; ++y) {
                I0[c * (H + 2 * P) * (W + 2 * P) + x * (W + 2 * P) + y] = (x < P || x >= H + P || y < P || y >= W + P) ? 0.0 : I[c * H * W + (x - P) * W + (y - P)];
            }
        }
    }

    double *d_I0, *d_F, *d_O;
    cudaMalloc(&d_I0, sizeof(double) * nI0);
    cudaMalloc(&d_F, sizeof(double) * nF);
    cudaMalloc(&d_O, sizeof(double) * nO);

    cudaMemcpy(d_I0, I0, sizeof(double) * nI0, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, F, sizeof(double) * nF, cudaMemcpyHostToDevice);

    dim3 blockDim(16, 16);
    dim3 gridDim((W + blockDim.x - 1) / blockDim.x, (H + blockDim.y - 1) / blockDim.y, K);

    cudaDeviceSynchronize();
    struct timespec start, end;
    double total_time = 0.0;

    clock_gettime(CLOCK_MONOTONIC, &start);
    basic_convolution<<<gridDim, blockDim>>>(d_I0, d_F, d_O);
    cudaDeviceSynchronize();
    clock_gettime(CLOCK_MONOTONIC, &end);

    total_time = (end.tv_sec - start.tv_sec) * 1000 + (end.tv_nsec - start.tv_nsec) / (double)MILLION;

    double *O = (double*)malloc(nO * sizeof(double));
    cudaMemcpy(O, d_O, sizeof(double) * nO, cudaMemcpyDeviceToHost);

    double checksum = 0.0;
    for (int k = 0; k < K; ++k) {
        for (int x = 0; x < W; ++x) {
            for (int y = 0; y < H; ++y) {
                checksum += O[k * W * H + x * H + y];
            }
        }
    }
    printf("C2_checksum: %.6e\n", checksum);
    printf("C2_execution_time: %.3f ms\n", total_time);
    free(I);
    free(F);
    free(I0);
    free(O);
    cudaFree(d_I0);
    cudaFree(d_F);
    cudaFree(d_O);

    return 0;
}
