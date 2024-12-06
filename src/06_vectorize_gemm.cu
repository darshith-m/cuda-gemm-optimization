#include <cuda_runtime.h>

template <const int BM, const int BN, const int BK, const int TM, const int TN>

__global__ void gemmVec2dBlockTiling (int M, int N, int K, 
                        float alpha,
                        float *A,
                        float *B,
                        float beta,
                        float *C) {
        
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
    const uint threadRowA = threadIdx.x / (BK / 4);
    const uint threadColA = threadIdx.x % (BK / 4);
    const uint threadRowB = threadIdx.x / (BN / 4);
    const uint threadColB = threadIdx.x % (BN / 4);

    // Allocate local registers for output matrix
    float threadResult[TM * TN] = {0.0};

    // Allocate local registers for As and Bs
    float regA[TM] = {0.0};
    float regB[TN] = {0.0};

    for (int blkIdx = 0; blkIdx < K; blkIdx += BK) {
        
        float4 temp = \
        reinterpret_cast<float4 *> (&A[threadRowA * K + threadColA * 4])[0];

        As[(threadColA * 4 + 0) * BM + threadRowA] = temp.x;
        As[(threadColA * 4 + 1) * BM + threadRowA] = temp.y;
        As[(threadColA * 4 + 2) * BM + threadRowA] = temp.z;
        As[(threadColA * 4 + 3) * BM + threadRowA] = temp.w;
        
        reinterpret_cast<float4 *>(&Bs[threadRowB * BN + threadColB * 4])[0] \
        = reinterpret_cast<float4 *>(&B[threadRowB * N + threadColB * 4])[0];

        __syncthreads();

        // Increment starting locations of cached blocks
        A += BK;
        B += BK * N;

        // Perform per-thread operation
        for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
            
            // Perform per-thread operation
            for (uint dotIdx = 0; dotIdx < BK; dotIdx++) {
                
                // Load shared memory block into local registers
                for (uint i = 0; i < TM; i++) {
                    regA[i] = As[dotIdx * BM + threadRow * TM + i];
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
        }
        
        for (uint resIdxA = 0; resIdxA < TM; resIdxA++) {
            for (uint resIdxB = 0; resIdxB < TN; resIdxB++) {
                
                float4 temp = reinterpret_cast<float4 *>\
                (&C[(threadRow * TM + resIdxA) * N + threadCol * TN + resIdxB])[0];
                
                temp.x = alpha * threadResult[resIdxA * TN + resIdxB] + beta * temp.x; 
                temp.y = alpha * threadResult[resIdxA * TN + resIdxB + 1] + beta * temp.y; 
                temp.z = alpha * threadResult[resIdxA * TN + resIdxB + 2] + beta * temp.z; 
                temp.w = alpha * threadResult[resIdxA * TN + resIdxB + 3] + beta * temp.w; 
                
                reinterpret_cast<float4 *>( \
                    &C[(threadRow * TM + resIdxA) * N + threadCol * TN + resIdxB])[0] = temp;
            }
        }
    }

}