#pragma once

#include "matrix_ops.h"
#include <cuda_runtime.h>

// Forward declarations of kernel functions
template<typename T>
__global__ void convolutionKernel(const T* input, const T* filter, T* output,
    int height, int width, int in_channels, int out_channels,
    int kernel_size, int padding, int stride);

template<typename T>
__global__ void tiledConvolutionKernel(const T* input, const T* filter, T* output,
    int height, int width, int in_channels, int out_channels,
    int kernel_size, int padding, int stride);

template<typename T>
class ConvolutionBase : public MatrixOperation<T> {
protected:
    int in_channels;
    int out_channels;
    int kernel_size;
    int padding;
    int stride;

    // Shared memory tile size for optimized implementation
    static constexpr int TILE_SIZE = 16;

public:
    ConvolutionBase(int h, int w, int in_c, int out_c, int k_size, 
                   int pad = 1, int str = 1) :
        MatrixOperation<T>(h, w, 1),
        in_channels(in_c), out_channels(out_c),
        kernel_size(k_size), padding(pad), stride(str) {}

    bool validateDimensions() const override {
        return this->rows > 0 && this->cols > 0 && 
               in_channels > 0 && out_channels > 0 && 
               kernel_size > 0 && kernel_size % 2 == 1;
    }
};

// Basic convolution implementation
template<typename T>
class BasicConvolution : public ConvolutionBase<T> {
public:
    using ConvolutionBase<T>::ConvolutionBase;
    
    void execute() override {
        dim3 blockDim(16, 16);
        dim3 gridDim(
            (this->cols + blockDim.x - 1) / blockDim.x,
            (this->rows + blockDim.y - 1) / blockDim.y,
            this->out_channels
        );

        convolutionKernel<<<gridDim, blockDim>>>(
            this->d_input1, this->d_input2, this->d_output,
            this->rows, this->cols, this->in_channels, this->out_channels,
            this->kernel_size, this->padding, this->stride
        );
    }
};

// Optimized convolution with shared memory
template<typename T>
class TiledConvolution : public ConvolutionBase<T> {
public:
    using ConvolutionBase<T>::ConvolutionBase;
    
    void execute() override {
        dim3 blockDim(this->TILE_SIZE, this->TILE_SIZE);
        dim3 gridDim(
            (this->cols + this->TILE_SIZE - 1) / this->TILE_SIZE,
            (this->rows + this->TILE_SIZE - 1) / this->TILE_SIZE,
            this->out_channels
        );

        tiledConvolutionKernel<<<gridDim, blockDim>>>(
            this->d_input1, this->d_input2, this->d_output,
            this->rows, this->cols, this->in_channels, this->out_channels,
            this->kernel_size, this->padding, this->stride
        );
    }
};