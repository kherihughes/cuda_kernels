# Part B: CUDA Memory Management Study

This section analyzes different memory management approaches in CUDA, comparing CPU, traditional GPU memory, and unified memory performance.

## Implementations

### CPU Implementation (q1)
- Baseline CPU vector addition
- Used for performance comparison
- Single-threaded implementation

### GPU Without Unified Memory (q2)
- Traditional CUDA memory management
- Explicit memory transfers using cudaMemcpy
- Tests different thread/block configurations

### GPU With Unified Memory (q3)
- Uses CUDA Unified Memory (cudaMallocManaged)
- Automatic memory management
- Same computational patterns as q2

## Thread Configurations Tested
1. Single block, single thread
2. Single block, 256 threads
3. Multiple blocks, 256 threads per block

## Usage

```bash
./q1 <K>  # CPU implementation
./q2 <K>  # GPU without unified memory
./q3 <K>  # GPU with unified memory
```
Where K is the size in millions of elements.

## Analysis Results

- Thread/Block configuration impact
- Memory transfer overhead analysis
- Unified vs non-unified memory comparison

See main [RESULTS.md](../RESULTS.md) for detailed performance data.