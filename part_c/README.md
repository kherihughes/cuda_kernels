# Part C: Convolution Implementations

This section implements and compares different approaches to 2D convolution on GPU.

## Implementations

### Basic Convolution (c1)
- Direct CUDA implementation
- No shared memory optimization
- Global memory access for each element

### Tiled Convolution (c2)
- Uses shared memory for input tiles
- Optimized memory access patterns
- Thread block synchronization for tile loading

### cuDNN Convolution (c3)
- Leverages NVIDIA's cuDNN library
- Highly optimized implementation
- Automatic algorithm selection

### Triton Implementation (c4, Bonus)
- Uses OpenAI's Triton framework
- Modern approach to GPU kernel optimization
- Demonstrates Triton's programming model

## Parameters
- Input size: 1024x1024
- Channels: 3
- Filter size: 3x3
- Number of filters: 64

## Usage

```bash
./c1  # Basic CUDA convolution
./c2  # Tiled optimization
./c3  # cuDNN implementation
```

## Performance Results

| Implementation | Time (ms) | Checksum |
|----------------|-----------|----------|
| Basic (C1) | 183.841 | 1.227563e+14 |
| Tiled (C2) | 3.055 | 1.165321e+14 |
| cuDNN (C3) | 0.562 | 1.227563e+14 |

See main [RESULTS.md](../RESULTS.md) for detailed analysis.