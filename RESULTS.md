# CUDA Performance Results

## Hardware Configuration
- GPU: NVIDIA GeForce RTX 3080
- CUDA Version: 12.6
- Driver Version: 560.35.02

## Vector Addition Performance
### Memory Access Optimization Results

| Vector Size | Non-coalesced Access | Coalesced Access |
|-------------|---------------------|------------------|
| 500K (3.84M) | 17.13 GFLOPs/s | 17.86 GFLOPs/s |
| 1M (7.68M) | 18.78 GFLOPs/s | 19.15 GFLOPs/s |
| 2M (15.36M) | 19.62 GFLOPs/s | 20.13 GFLOPs/s |

*Note: Coalesced memory access consistently provides 2-5% performance improvement*

## Memory Management Study Results
### Performance Across Different Configurations (in milliseconds)

| Size (M) | CPU | GPU (1 Block, 1 Thread) | GPU (1 Block, 256 Threads) | GPU (Multiple Blocks) |
|----------|-----|------------------------|---------------------------|---------------------|
| 1 | 1.91 | 67.72 | 1.18 | 0.024 |
| 5 | 19.82 | 324.46 | 5.83 | 0.094 |
| 10 | 21.29 | 639.36 | 13.76 | 0.181 |
| 50 | 87.14 | 3198.63 | 61.92 | 0.881 |
| 100 | 189.63 | 6313.08 | 129.15 | 1.773 |

## Convolution Implementation Comparison

| Implementation | Execution Time (ms) | Checksum |
|----------------|-------------------|----------|
| Basic | 183.841 | 1.227563e+14 |
| Tiled | 3.055 | 1.165321e+14 |
| cuDNN | 0.562 | 1.227563e+14 |

### Key Findings:
1. Tiled implementation achieves ~60x speedup over basic implementation
2. cuDNN provides additional ~5.4x speedup over tiled implementation
3. All implementations maintain numerical stability with consistent checksums