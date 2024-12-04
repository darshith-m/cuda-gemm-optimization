#include <cuda_runtime.h>

/*
Matrix dimensions:
A = M x K
B = K x N
C = M x N
*/

template <const uint BLOCK_SIZE>
__global__ void gemmMemCoalesced(int M, int N, int K,
                         float alpha,
                         const float *A,
                         const float *B,
                         float beta,
                         float *C) {
    
    // Memory coalesced thread indexing
    const uint x = blockIdx.x * BLOCK_SIZE + (threadIdx.x / BLOCK_SIZE);
    const uint y = blockIdx.y * BLOCK_SIZE + (threadIdx.x % BLOCK_SIZE);

    if ((x < M) && (y < N)) {
        float temp = 0.0;
        for (int i = 0; i < K; i++) {
            temp += A[x * K + i] * B[i * N + y];
        }
        __syncthreads();
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}