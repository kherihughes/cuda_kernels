# Part A: Vector Addition and Matrix Multiplication Optimizations

This section implements and compares different optimization techniques for basic CUDA operations.

## Vector Addition

### Non-coalesced Implementation (vecaddKernel00)
- Basic implementation without memory coalescing
- Each thread processes multiple elements sequentially
- Demonstrates baseline performance

### Coalesced Implementation (vecaddKernel01)
- Optimized implementation with coalesced memory access
- Adjacent threads access adjacent memory locations
- Shows impact of proper memory access patterns

## Matrix Multiplication

### Basic Implementation (matmultKernel00)
- Standard matrix multiplication algorithm
- Uses shared memory for tile-based computation
- Basic synchronization between threads

### Optimized Implementation (matmultKernel01)
- Uses shared memory with optimized access patterns
- Implements loop unrolling for better instruction throughput
- Each thread computes multiple output elements

## Usage

```bash
# Vector Addition
./vecadd00 <size>  # Non-coalesced
./vecadd01 <size>  # Coalesced

# Matrix Multiplication
./matmult00 <size>  # Basic
./matmult01 <size>  # Optimized
```

## Performance Results

See main [RESULTS.md](../RESULTS.md) for detailed performance analysis.