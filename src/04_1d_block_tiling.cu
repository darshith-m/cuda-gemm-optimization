#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM>

__global__ void gemm1dBlockTiling (int M, int N, int K, 
                        float alpha,
                        const float *A,
                        const float *B,
                        float beta,
                        float *C) {
    
    // The output block being computed
    const uint row = blockIdx.x;
    const uint col = blockIdx.y;

    // Allocate Shared Memory space
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Assign thread row and col indices in the block
    const uint threadRow = threadIdx.x / BN;
    const uint threadCol = threadIdx.x % BN;

    // Move base address of the matrices
    A += row * BM * K;
    B += col * BN;
    C += row * BM * N + col * BN;

    // Warp-level Global Memory coalescing 
    const uint threadRowA = threadIdx.x / BK;
    const uint threadColA = threadIdx.x % BK;
    const uint threadRowB = threadIdx.x / BN;
    const uint threadColB = threadIdx.x % BN;

    // Allocate local registers for output matrix
    float threadResult[TM] = {0.0};

    for (uint blkIdx = 0; blkIdx < K; blkIdx += BK) {

        // Assign Shared Memory
        As[threadRowA * BK + threadColA] = A[threadRowA * K + threadColA];
        Bs[threadRowB * BN + threadColB] = B[threadRowB * N + threadColB];

        __syncthreads();

        // Increment starting locations of cached blocks
        A += BK;
        B += BK * N;

        // Perform per-thread operation
        for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
            // Store Bs matrix in local register for re-use
            float tempBs = Bs[dotIdx * BN + threadCol];
            for (uint resIdx = 0; resIdx < TM; resIdx++) {
                threadResult[resIdx] += As[(threadRow * TM + resIdx) * BK + dotIdx] * tempBs;
            }
        }
        __syncthreads();
    }

    for (uint resIdx = 0; resIdx < TM; resIdx++)  {
        C[(threadRow * TM + resIdx) * N + threadCol] \
        = alpha * threadResult[resIdx] \
        + beta * C[(threadRow * TM + resIdx) * N + threadCol];
    }
}