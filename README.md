# High-Performance CUDA Optimizations

This repository demonstrates various CUDA optimization techniques for high-performance computing operations, including vector addition, matrix multiplication, memory management analysis, and convolution implementations.

**Motivation:** This project aims to explore and implement key optimization strategies in CUDA for fundamental linear algebra and convolution operations. By comparing the performance of different implementations, we seek to gain a deeper understanding of the impact of techniques like coalescing, shared memory usage, algorithm choice, and memory management strategies on overall GPU performance.

## Optimizations Implemented

This repository is divided into three main sections, each focusing on different aspects of CUDA optimization:

### Vector and Matrix Operations
-   **Vector Addition:** Implementations using both non-coalesced (`vecadd00`) and coalesced (`vecadd01`) memory access patterns. We demonstrate how coalescing can improve performance by reducing memory transactions and improving memory bandwidth utilization.
-   **Matrix Multiplication:** Implementations include a basic version using shared memory (`matmult00`) and an optimized version that utilizes shared memory with coalesced access and loop unrolling (`matmult01`). This highlights the importance of these optimization techniques for compute-bound operations, focusing on reducing redundant memory access and maximizing instruction throughput.

### Memory Management Study
-   **CPU vs GPU Performance:** Benchmarking of vector addition on the CPU (`q1`), and on the GPU using both traditional CUDA memory allocation (`q2`) and Unified Memory (`q3`). These benchmarks compare performance, explore different configurations and analyze the overhead of explicit data transfers versus unified memory access.
-   **Thread/Block Configuration:** We experiment with various thread/block configurations (single block, single thread, single block with multiple threads, multiple blocks with multiple threads) to identify optimal configurations for the GPU, emphasizing the impact of different thread/block organizations on memory access patterns and workload distribution.
-   **Unified vs Non-Unified Memory:** We analyze the impact of CUDA Unified Memory on the performance of simple vector addition tasks, comparing how using a single, unified address space can ease development and potentially improve performance over explicit memory transfers.

### Convolution Implementations
-   **Basic CUDA Convolution:** A straightforward implementation of 2D convolution in CUDA (`c1`), serving as a baseline for performance comparison and algorithm verification.
-   **Tiled Convolution:** A more optimized implementation (`c2`) utilizing shared memory and tiling to reduce global memory transactions, enhance data reuse and improve memory coalescing, thereby improving overall memory bandwidth utilization.
-   **cuDNN Convolution:** Leveraging NVIDIA's cuDNN library for highly optimized convolution performance (`c3`). Demonstrates the power of using highly optimized, vendor-supplied libraries and their ability to auto-tune for optimal performance.
-   **Triton Framework Implementation (Bonus):** A custom implementation of convolution using OpenAI's Triton framework (`c4`). This provides an example of how to define and utilize a custom kernel for convolution, showcasing a modern and flexible approach to GPU programming for deep learning.

## Requirements

-   CUDA Toolkit >= 11.7 (Tested with versions 11.7 and 12.6)
-   cuDNN (Required for Part C's cuDNN implementation)
-   CMake >= 3.10
-   C++14 capable compiler
-   Python 3.8+ (For analysis, visualization and plotting)
-   GPU with compute capability >= 7.5 (Required for advanced compute features and shared memory size)

## Building the Project

1.  Clone this repository.
2.  Create a build directory:
    ```bash
    mkdir build && cd build
    ```
3.  Run CMake to configure the project, creating the necessary build files:
    ```bash
    cmake ..
    ```
4.  Build the project using `make`, leveraging multiple cores for a faster build:
    ```bash
    make -j4
    ```

## Running Tests & Benchmarks

**Quick Tests:**
   ```bash
   ./scripts/run_tests.sh
   ```
   This script compiles the project (if a build folder does not exist) and runs basic verification tests for each part of the project to check for build issues and basic functionality.

**Full Benchmarks:**
    ```bash
    ./scripts/run_benchmarks.sh
    ```
    This script executes comprehensive benchmarks for all parts of the project. It will compile (if no build folder exists) and then generate more detailed results.

## Performance Results

Detailed performance results, including analysis and observations, are available in [RESULTS.md](./RESULTS.md). Below is a quick summary highlighting key findings:

### Vector Addition Performance (GFLOPs/s) - Part A
| Size | Non-coalesced | Coalesced |
|------|--------------|-----------|
| 500K | 17.13 | 17.86 |
| 1M | 18.78 | 19.15 |
| 2M | 19.62 | 20.13 |
*   **Coalescing:** Vector addition with coalesced memory access consistently outperforms the non-coalesced version by ~2-5%, showcasing the significance of aligning memory access patterns with GPU memory architecture.

### Convolution Performance (ms) - Part C

| Implementation | Time | Checksum |
|---------------|------|----------|
| Basic | 183.841 | 1.227563e+14 |
| Tiled | 3.055 | 1.165321e+14 |
| cuDNN | 0.562 | 1.227563e+14 |

*   **Tiling:** Tiled convolution using shared memory (C2) achieves a remarkable ~60x speedup over the basic implementation (C1), highlighting the efficiency gains from reducing global memory access.
*   **cuDNN:** The cuDNN implementation (C3) provides an additional ~5.4x speedup over the tiled implementation, showcasing the optimizations available through specialized libraries.

See [RESULTS.md](./RESULTS.md) for a complete and detailed performance analysis of each optimization, including more configurations and results for different problem sizes and analysis of memory management results.

## Repository Structure
```
.
├── part_a/          # Vector and matrix operations
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── include
│   │   ├── matmultKernel.h
│   │   └── vecaddKernel.h
│   └── src
│       ├── matmult.cu
│       ├── matmultKernel00.cu
│       ├── matmultKernel01.cu
│       ├── vecadd.cu
│       ├── vecaddKernel00.cu
│       └── vecaddKernel01.cu
├── part_b/          # Memory management study
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── results
│   │   └── Figure_1.png
│   └── src
│       ├── q1.cpp
│       ├── q2.cu
│       ├── q3.cu
│       └── q4.py
├── part_c/          # Convolution implementations
│   ├── CMakeLists.txt
│   ├── README.md
│   ├── include
│   │   └── convolution_impl.h
│   ├── program_output.csv
│   ├── results
│   └── src
│       ├── c1.cu
│       ├── c2.cu
│       ├── c3.cu
│       ├── c4 copy.ipynb
│       └── c4.ipynb
├── common/          # Shared utilities
│   ├── CMakeLists.txt
│   └── timer
│       ├── timer.cu
│       └── timer.h
├── include/         # Common headers
│   ├── convolution_kernels.h
│   ├── convolution_ops.h
│   ├── cuda_utils.h
│   └── matrix_ops.h
├── scripts/         # Build and test scripts
│    ├── collect_repo.sh
│    ├── run_benchmarks.sh
│    └── run_tests.sh
├── requirements.txt  # Python requirements
├── build.sh         # Build automation
├── CMakeLists.txt   # Main CMake config file
├── README.md        # This README file
└── RESULTS.md        # Performance Results file
```

**Part A:** Contains CUDA implementations and executables for vector and matrix operations.
  *   `src/` Includes `.cu` files for kernels and main functions.
  *   `include/` Includes header files for kernels.

**Part B:** Contains code related to the performance study of Unified Memory and different configurations.
   * `src/` Includes `.cu` and `.cpp` files.
   * `scripts/` includes python for visualization

**Part C:** Contains source code for implementations of convolution operations.
   * `src/` Includes `.cu` files for kernels and main functions.
    * `include/` Includes header files and for the kernels and structs for conv operations.

**common:** Contains generic and reusable utilities.
   * `timer/` Includes source and header file for timing utilities.

**include:** Contains generic and reusable headers
   *  includes headers used in multiple parts of the project.

**scripts:** Contains convenience bash scripts for building, running tests and performing benchmark operations.

**docs:** Intended for future documentation efforts for the project but currently unused.

**requirements.txt:** List of Python packages to install for analyzing and visualizing data.

**build.sh:** A convenience bash script used by other bash scripts to build the project.

**CMakeLists.txt:** The root CMake file for the project.

**README.md:** This documentation you're reading.

**RESULTS.md:** File with all the performance results and analysis.

## Author

Kheri Hughes