#pragma once

#include "cuda_utils.h"

template<typename T>
class MatrixOperation {
protected:
    T* d_input1;  // First input matrix
    T* d_input2;  // Second input matrix
    T* d_output;  // Output matrix
    int rows;
    int cols;
    int batch_size;

    // Common initialization logic
    void initializeDeviceMemory(size_t size1, size_t size2, size_t size_out) {
        cuda_utils::allocateDeviceMemory(&d_input1, size1);
        cuda_utils::allocateDeviceMemory(&d_input2, size2);
        cuda_utils::allocateDeviceMemory(&d_output, size_out);
    }

    // Common cleanup logic
    void cleanup() {
        cuda_utils::freeDeviceMemory(d_input1);
        cuda_utils::freeDeviceMemory(d_input2);
        cuda_utils::freeDeviceMemory(d_output);
    }

public:
    MatrixOperation(int r, int c, int b = 1) : 
        rows(r), cols(c), batch_size(b), 
        d_input1(nullptr), d_input2(nullptr), d_output(nullptr) {}

    virtual ~MatrixOperation() {
        cleanup();
    }

    // Pure virtual function for executing the operation
    virtual void execute() = 0;

    // Common validation check
    virtual bool validateDimensions() const = 0;

    // Memory transfer helpers
    void copyInputToDevice(const T* h_input1, const T* h_input2, 
                          size_t size1, size_t size2) {
        cuda_utils::copyToDevice(d_input1, h_input1, size1);
        cuda_utils::copyToDevice(d_input2, h_input2, size2);
    }

    void copyOutputToHost(T* h_output, size_t size) {
        cuda_utils::copyToHost(h_output, d_output, size);
    }
};