#include <cuda_runtime.h>

template <const int BLOCK_SIZE>

__global__ void gemmSharedMem (int M, int N, int K, 
                        float alpha,
                        const float *A,
                        const float *B,
                        float beta,
                        float *C) {
    
    // The output block being computed
    const uint row = blockIdx.x;
    const uint col = blockIdx.y;

    // Allocate Shared Memory space
    __shared__ float As[BLOCK_SIZE * BLOCK_SIZE];
    __shared__ float Bs[BLOCK_SIZE * BLOCK_SIZE];

    // Assign thread row and col indices in the block
    const uint threadRow = threadIdx.x / BLOCK_SIZE;
    const uint threadCol = threadIdx.x % BLOCK_SIZE;

    // Move base address of the matrices
    A += row * BLOCK_SIZE * K;
    B += col * BLOCK_SIZE;
    C += row * BLOCK_SIZE * N + col * BLOCK_SIZE;

    float temp = 0.0;
    for (int blkIdx = 0; blkIdx < K; blkIdx += BLOCK_SIZE) {

        As[threadRow * BLOCK_SIZE + threadCol] = A[threadRow * K + threadCol];
        Bs[threadRow * BLOCK_SIZE + threadCol] = B[threadRow * K + threadCol];

        __syncthreads();

        A += BLOCK_SIZE;
        B += BLOCK_SIZE * N;

        // Execute dot product of blocked matrix
        for (int k = 0; k < BLOCK_SIZE; k++) {
            temp += As[threadRow * BLOCK_SIZE + k] \
            * Bs[k * BLOCK_SIZE + threadCol];
        }

        __syncthreads();

        C[threadRow * N + threadCol] = alpha * temp + beta * C[threadRow * N + threadCol];

        __syncthreads();

    }

}