# CUDA GEMM Optimization

## GEMM (General Matrix Multiplication)

GEMM (General Matrix Multiplication) is a fundamental matrix operation (C = A × B) that's critical in modern computing, especially for AI/ML. It's the core computation in neural networks, making it essential for both training and inference.

Key importance:
- Powers deep learning models through efficient matrix operations
- Hardware (GPUs, TPUs) and libraries (BLAS, cuBLAS) are optimized for it
- Determines energy efficiency in AI computing
- Used in graphics, scientific computing, and quantum simulations

Its performance directly impacts AI model training speed and inference efficiency, making it a key benchmark for computing platforms.

## CUDA Optimizations

Memory Coalescing:
- Ensures adjacent threads access adjacent memory locations
- Reduces memory transactions by bundling them together
- Key for maximizing memory bandwidth utilization

Shared Memory:
- Low-latency on-chip memory shared within thread block
- Reduces global memory accesses
- Used as scratchpad for intermediate results
- Handles bank conflicts through proper padding

1D Tiling:
- Loads data strip into shared memory
- Increases data reuse across thread block
- Reduces global memory bandwidth pressure
- Better cache utilization

2D Tiling:
- Extends 1D concept to both matrix dimensions
- Square tiles maximize data reuse
- Further reduces global memory accesses
- Optimal tile size depends on GPU architecture

Vectorization:
- Uses vector loads/stores (float4, float2)
- Increases memory throughput
- Better instruction-level parallelism
- Reduces instruction count
- Can improve occupancy through register reduction

These optimizations combined give significant speedups in GEMM operations, compared to naive implementations.

## Project Directory

.
├── .vscode/          # VS Code configuration
├── images/           # Documentation images and diagrams
├── profiles/         # Performance profiling data (Nsight Compute reports)
├── src/              # Source code files
├── gemm_optimization.ipynb   # Jupyter notebook with optimization steps
└── README.md         # Project introduction

# CUDA GEMM Optimization Project

A CUDA-based matrix multiplication implementation showcasing progressive optimization techniques.

## Directory Structure
```
.
├── .vscode/             # Editor config
├── images/             # GEMM visualization
├── profiles/          # Nsight Compute metrics
├── src/              # CUDA source files
├── gemm_optimization.ipynb   # Main notebook
└── README.md         # Documentation
```

### Hardware Setup

GPU: NVIDIA RTX 3060 Mobile
VRAM: 6GB GDDR6

### Performance Study
This project examines various GEMM optimizations for square matrices of size 4096x4096. The implementations progress from naive CUDA to optimized versions using techniques like memory coalescing, shared memory tiling, and vectorization. Each optimization's performance is analyzed using Nsight Compute metrics including memory bandwidth utilization, cache hit rates, and SM occupancy.

### Usage

1. Execute notebook cells sequentially
2. Each optimization technique analyzed:
    - Memory coalescing
    - Shared memory utilization
    - 1D/2D tiling
    - Vectorization

3. Analysis
Use Nsight Compute for:
- Performance profiling
- Memory bandwidth
- Occupancy metrics
- Cache behavior

Follow notebook for step-by-step optimization analysis.

### Results

## Performance Results

| Optimization | Performance (GFLOP/s) | Arithmetic Intensity (FLOP/byte) |
|--------------|---------------------|--------------------------------|
| Naive CUDA | 51.8 | 14.58 |
| Global Memory Coalescing | 374.9 | 15.14 |
| Shared Memory | 505.0 | 15.85 |
| 1D Tiling | 1,603.9 | 28.46 |
| 2D Tiling | 248.7 | 4.76 |
| 2D Tiling + Vectorization | 3,824.1 | 58.25 |

The best performance is achieved with 2D tiling combined with vectorization, showing ~74x speedup over naive implementation.

Need to resolve shared memory bank conflicts!!