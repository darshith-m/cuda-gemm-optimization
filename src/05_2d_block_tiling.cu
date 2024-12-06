#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>

__global__ __launch_bounds__((BM * BN) / (TM * TN), 1) 
    void gemm2dBlockTiling (int M, int N, int K, 
                        float alpha,
                        float *A,
                        float *B,
                        float beta,
                        float *C) {
    
    
    const uint totalNumResultsPerBlock = BM * BN;
    const uint numThreadsPerBlock = totalNumResultsPerBlock / (TM * TN);
    
    // The output block being computed
    const uint row = blockIdx.y;
    const uint col = blockIdx.x;

    // Allocate Shared Memory space
    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    // Assign thread row and col indices in the block
    const uint threadRow = threadIdx.x / (BN/TN);
    const uint threadCol = threadIdx.x % (BN/TN);

    // Move base address of the matrices
    A += row * BM * K;
    B += col * BN;
    C += row * BM * N + col * BN;

    // Warp-level Global Memory coalescing 
    const uint threadRowA = threadIdx.x / BK;
    const uint threadColA = threadIdx.x % BK;
    const uint threadRowB = threadIdx.x / BN;
    const uint threadColB = threadIdx.x % BN;

    const uint strideA = numThreadsPerBlock / BK;
    const uint strideB = numThreadsPerBlock / BN;

    // Allocate local registers for output matrix
    float threadResult[TM * TN] = {0.0};

    // Allocate local registers for As and Bs
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    for (int blkIdx = 0; blkIdx < K; blkIdx += BK) {

        // Assign Shared Memory
        for (uint offset = 0; offset < BM; offset += strideA) {
            As[(threadRowA + offset) * BK + threadColA] = \
            A[(threadRowA + offset) * K + threadColA];
        }

        for (uint offset = 0; offset < BK; offset += strideB) {
            Bs[(threadRowB + offset) * BN + threadColB] = \
            B[(threadRowB + offset) * N + threadColB];
        }

        __syncthreads();

        // Increment starting locations of cached blocks
        A += BK;
        B += BK * N;

        // Perform per-thread operation
        for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
            
            // Load shared memory block into local registers
            for (uint i = 0; i < TM; i++) {
                regA[i] = As[(threadRow * TM + i) * BK + dotIdx];
            }
            for (uint i = 0; i < TN; i++) {
                regB[i] = Bs[dotIdx * BN + threadCol * TN + i];
            }

            for (uint resIdxA = 0; resIdxA < TM; resIdxA++) {
                for (uint resIdxB = 0; resIdxB < TN; resIdxB++) {
                    threadResult[resIdxA * TN + resIdxB] \
                    += regA[resIdxA] * regB[resIdxB];
                }
            }
            __syncthreads();
        }

        for (uint resIdxA = 0; resIdxA < TM; resIdxA++) {
            for (uint resIdxB = 0; resIdxB < TN; resIdxB++) {
                C[(threadRow * TM + resIdxA) * N + threadCol * TN + resIdxB] \
                = alpha * threadResult[resIdxA * TN + resIdxB] \
                + beta * C[(threadRow * TM + resIdxA) * N + threadCol * TN + resIdxB];
            }
        }
        __syncthreads();
    }

}