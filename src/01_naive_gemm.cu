#include <cuda_runtime.h>

/*
Matrix dimensions:
A = M x K
B = K x N
C = M x N
*/
__global__ void gemmNaive(int M, int N, int K, 
                        float alpha,
                        const float *A,
                        const float *B,
                        float beta,
                        float *C) {

    const uint x = blockIdx.x * blockDim.x + threadIdx.x;
    const uint y = blockIdx.y * blockDim.y + threadIdx.y;

    if ((x < M) && (y < N)) {
        float temp = 0.0;
        for (int k = 0; k < K; k++) {
            temp += A[x * K + k] * B[k * N + y];
        }
        __syncthreads();
        C[x * N + y] = alpha * temp + beta * C[x * N + y];
    }
}