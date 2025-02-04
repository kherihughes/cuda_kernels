#pragma once

#include <cuda_runtime.h>

template<typename T>
__global__ void convolutionKernel(const T* input, const T* filter, T* output,
    int height, int width, int in_channels, int out_channels,
    int kernel_size, int padding, int stride) {
    
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int k = blockIdx.z;

    if (x < width && y < height) {
        T sum = 0;
        for (int c = 0; c < in_channels; ++c) {
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    int x_in = x + i - padding;
                    int y_in = y + j - padding;
                    
                    if (x_in >= 0 && x_in < width && y_in >= 0 && y_in < height) {
                        int input_idx = c * height * width + y_in * width + x_in;
                        int filter_idx = k * in_channels * kernel_size * kernel_size + 
                                       c * kernel_size * kernel_size + 
                                       i * kernel_size + j;
                        sum += input[input_idx] * filter[filter_idx];
                    }
                }
            }
        }
        output[k * height * width + y * width + x] = sum;
    }
}

template<typename T>
__global__ void tiledConvolutionKernel(const T* input, const T* filter, T* output,
    int height, int width, int in_channels, int out_channels,
    int kernel_size, int padding, int stride) {
    
    extern __shared__ T shared_mem[];
    
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int x = blockIdx.x * blockDim.x + tx;
    int y = blockIdx.y * blockDim.y + ty;
    int k = blockIdx.z;

    // Load data into shared memory
    if (x < width && y < height) {
        for (int c = 0; c < in_channels; ++c) {
            shared_mem[(c * blockDim.y + ty) * blockDim.x + tx] = 
                input[c * height * width + y * width + x];
        }
    }
    
    __syncthreads();

    // Compute convolution
    if (x < width && y < height) {
        T sum = 0;
        for (int c = 0; c < in_channels; ++c) {
            for (int i = 0; i < kernel_size; ++i) {
                for (int j = 0; j < kernel_size; ++j) {
                    int x_in = tx + i - padding;
                    int y_in = ty + j - padding;
                    
                    if (x_in >= 0 && x_in < blockDim.x && y_in >= 0 && y_in < blockDim.y) {
                        int shared_idx = (c * blockDim.y + y_in) * blockDim.x + x_in;
                        int filter_idx = k * in_channels * kernel_size * kernel_size + 
                                       c * kernel_size * kernel_size + 
                                       i * kernel_size + j;
                        sum += shared_mem[shared_idx] * filter[filter_idx];
                    }
                }
            }
        }
        output[k * height * width + y * width + x] = sum;
    }
}