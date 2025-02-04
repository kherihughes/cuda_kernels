#include <stdio.h>
#include <stdlib.h>
#include <cudnn.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <iostream>
#include <stdlib.h>
#include <time.h>
#include <cstdio>
#include "include/cuda_utils.h"
#include "include/convolution_ops.h"

#define H 1024     // Input height
#define W 1024     // Input width
#define C 3        // Number of input channels
#define FH 3       // Filter height
#define FW 3       // Filter width
#define K 64       // Number of filters (output channels)
#define P 1        // Padding

// Error checking macro
#define checkCUDNN(expression)                               \
{                                                           \
    cudnnStatus_t status = (expression);                    \
    if (status != CUDNN_STATUS_SUCCESS) {                   \
        fprintf(stderr, "Error on line %d: %s\n",           \
                __LINE__, cudnnGetErrorString(status));     \
        exit(EXIT_FAILURE);                                 \
    }                                                       \
}

int main() {
    // Define tensor dimensions
    int input_n = 1;    // Batch size
    int input_c = C;
    int input_h = H;
    int input_w = W;

    int output_n = 1;
    int output_c = K;
    int output_h = H;   // Assuming stride=1 and padding=P
    int output_w = W;

    int filter_k = K;
    int filter_c = C;
    int filter_h = FH;
    int filter_w = FW;

    // Allocate host memory
    size_t input_size = input_n * input_c * input_h * input_w * sizeof(float);
    size_t filter_size = filter_k * filter_c * filter_h * filter_w * sizeof(float);
    size_t output_size = output_n * output_c * output_h * output_w * sizeof(float);

    float *h_I = (float *)malloc(input_size);
    float *h_F = (float *)malloc(filter_size);
    float *h_O = (float *)malloc(output_size);

    // Initialize input tensor I
    for (int c = 0; c < C; c++)
        for (int y = 0; y < H; y++)
            for (int x = 0; x < W; x++)
                h_I[(c * H + y) * W + x] = c * (x + y);

    // Initialize filter tensor F with reversed indices (pre-flip)
    for (int k = 0; k < K; k++)
        for (int c = 0; c < C; c++)
            for (int j = 0; j < FH; j++)
                for (int i = 0; i < FW; i++) {
                    int fi = FW - 1 - i;
                    int fj = FH - 1 - j;
                    h_F[(((k * C + c) * FH + fj) * FW) + fi] = (c + k) * (i + j);
                }

    // Allocate device memory
    float *d_I, *d_F, *d_O;
    cudaMalloc((void **)&d_I, input_size);
    cudaMalloc((void **)&d_F, filter_size);
    cudaMalloc((void **)&d_O, output_size);

    // Copy data to device memory
    cudaMemcpy(d_I, h_I, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_F, h_F, filter_size, cudaMemcpyHostToDevice);

    // Create cuDNN handle
    cudnnHandle_t cudnn;
    checkCUDNN(cudnnCreate(&cudnn));

    // Create tensor descriptors
    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        input_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        input_n,
        input_c,
        input_h,
        input_w
    ));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(
        output_descriptor,
        CUDNN_TENSOR_NCHW,
        CUDNN_DATA_FLOAT,
        output_n,
        output_c,
        output_h,
        output_w
    ));

    // Create filter descriptor
    cudnnFilterDescriptor_t filter_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&filter_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(
        filter_descriptor,
        CUDNN_DATA_FLOAT,
        CUDNN_TENSOR_NCHW,
        filter_k,
        filter_c,
        filter_h,
        filter_w
    ));

    // Create convolution descriptor
    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(
        convolution_descriptor,
        P, P,   // padding
        1, 1,   // stride
        1, 1,   // dilation
        CUDNN_CROSS_CORRELATION,
        CUDNN_DATA_FLOAT
    ));
    
    
    cudnnConvolutionFwdAlgo_t convolution_algorithm;

    // Choose the best convolution algorithm
    int returnedAlgoCount = 0;
    cudnnConvolutionFwdAlgoPerf_t algoPerf;
    checkCUDNN(cudnnFindConvolutionForwardAlgorithm(
        cudnn,
        input_descriptor,
        filter_descriptor,
        convolution_descriptor,
        output_descriptor,
        1, // request one algorithm
        &returnedAlgoCount,
        &algoPerf
    ));
    convolution_algorithm = algoPerf.algo;


    // Get workspace size
    size_t workspace_bytes = 0;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(
        cudnn,
        input_descriptor,
        filter_descriptor,
        convolution_descriptor,
        output_descriptor,
        convolution_algorithm,
        &workspace_bytes
    ));

    // Allocate workspace memory
    void *d_workspace = NULL;
    cudaMalloc(&d_workspace, workspace_bytes);

    // Set convolution scaling parameters
    float alpha = 1.0f;
    float beta = 0.0f;

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Perform convolution
    cudaEventRecord(start);
    checkCUDNN(cudnnConvolutionForward(
        cudnn,
        &alpha,
        input_descriptor,
        d_I,
        filter_descriptor,
        d_F,
        convolution_descriptor,
        convolution_algorithm,
        d_workspace,
        workspace_bytes,
        &beta,
        output_descriptor,
        d_O
    ));
    cudaEventRecord(stop);

    // Wait for convolution to finish
    cudaEventSynchronize(stop);

    // Calculate elapsed time
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // Copy result back to host
    cudaMemcpy(h_O, d_O, output_size, cudaMemcpyDeviceToHost);

    // Compute checksum
    double checksum = 0.0;
    for (int idx = 0; idx < output_n * output_c * output_h * output_w; idx++) {
        checksum += (double)h_O[idx];
    }

    // Print checksum and kernel execution time
    printf("C3_checksum: %.6e\n", checksum);
    printf("C3_execution_time: %.3f ms\n", milliseconds);

    // Clean up
    cudaFree(d_I);
    cudaFree(d_F);
    cudaFree(d_O);
    cudaFree(d_workspace);

    free(h_I);
    free(h_F);
    free(h_O);

    cudnnDestroyTensorDescriptor(input_descriptor);
    cudnnDestroyTensorDescriptor(output_descriptor);
    cudnnDestroyFilterDescriptor(filter_descriptor);
    cudnnDestroyConvolutionDescriptor(convolution_descriptor);
    cudnnDestroy(cudnn);

    return 0;
}