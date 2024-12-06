#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <time.h>
#include "nvtx3/nvToolsExt.h"

#include "./01_naive_gemm.cu"
#include "./02_memory_coalesced_gemm.cu"
#include "./03_shared_memory_gemm.cu"
#include "./04_1d_block_tiling.cu"
#include "./05_2d_block_tiling.cu"
#include "./06_vectorize_gemm.cu"

#define CEIL_DIV(x,y) (((x) + (y) - 1) / (y)) 
#define BLOCK_SIZE 32

#define CUDA_CHECK_ERROR(call) do { \
   cudaError_t err = call; \
   if (err != cudaSuccess) { \
       printf("CUDA error %s at line %d: %s\n", #call, __LINE__, cudaGetErrorString(err)); \
       exit(1); \
   } \
} while(0)

void randomInit(float *data, int size) {
    for (int i = 0; i < size; ++i)
        //data[i] = rand() / (float)RAND_MAX;
        data[i] = 1.0;
}

// Verification
bool verifyResults(float* gpu_cu, float* gpu, int size, float tolerance=1e-5) {
   for(int i = 0; i < size; i++) {
       if(fabs(gpu_cu[i] - gpu[i]) > tolerance) {
           printf("Mismatch at %d: cuBLAS=%f, Manual implementation=%f\n", i, gpu_cu[i], gpu[i]);
           return false;
       }
   }
   return true;
}

void runCublas(int m, int n, int k, float alpha, float* d_A, float* d_B, float beta, float* d_C_cublas) {
    // Run cuBLAS
    cublasHandle_t handle;
    cublasCreate(&handle);
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, 
                n, m, k,
                &alpha,
                d_B, n,
                d_A, k, 
                &beta,
                d_C_cublas, n);
    cublasDestroy(handle);
}

void runGemmNaive(int m, int n, int k, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    gemmNaive<<<gridSize, blockSize>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
}

void runGemmMemCoalesced(int m, int n, int k, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    gemmMemCoalesced<BLOCK_SIZE><<<gridSize, blockSize>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
}

void  runGemmSharedMem(int m, int n, int k, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    dim3 blockSize(BLOCK_SIZE * BLOCK_SIZE);
    dim3 gridSize(CEIL_DIV(m, 32), CEIL_DIV(n, 32));
    gemmSharedMem<BLOCK_SIZE><<<gridSize, blockSize>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
}

void  runGemm1dBlockTiling(int m, int n, int k, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    const uint BM = 64;
    const uint BN = 64;
    const uint BK = 8;
    const uint TM = 8; 
    dim3 blockSize((BM * BN) / TM);
    dim3 gridSize(CEIL_DIV(m, BM), CEIL_DIV(n, BN));
    gemm1dBlockTiling<BM, BN, BK, TM><<<gridSize, blockSize>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
}

void  runGemm2dBlockTiling(int m, int n, int k, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    dim3 blockSize((BM * BN) / (TM * TN));
    dim3 gridSize(CEIL_DIV(m, BN), CEIL_DIV(n, BM));
    gemm2dBlockTiling<BM, BN, BK, TM, TN><<<gridSize, blockSize>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
}

void  runGemmVec2dBlockTiling(int m, int n, int k, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
    const uint BM = 128;
    const uint BN = 128;
    const uint BK = 8;
    const uint TM = 8;
    const uint TN = 8;
    dim3 blockSize((BM * BN) / (TM * TN));
    dim3 gridSize(CEIL_DIV(m, BN), CEIL_DIV(n, BM));
    gemmVec2dBlockTiling<BM, BN, BK, TM, TN><<<gridSize, blockSize>>>(m, n, k, alpha, d_A, d_B, beta, d_C);
}

int main(int argc, char **argv) {

    cudaDeviceReset();

    nvtxRangePush("Matrix Multiplication");
     
    const float alpha = 1.0f;
    const float beta = 0.0f;

    if (argc != 5) {
        printf("Usage: %s <choice> <m> <n> <k>\n", argv[0]);
        return -1;
    }

    int x = atoi(argv[1]);
    int m = atoi(argv[2]);
    int n = atoi(argv[3]);
    int k = atoi(argv[4]);

    // Allocate host memory
    float *h_A = (float*)malloc(m * k * sizeof(float));
    float *h_B = (float*)malloc(k * n * sizeof(float));
    float *h_C = (float*)malloc(m * n * sizeof(float));
    float *h_C_cublas = (float*)malloc(m * n * sizeof(float));

    // Initialize host matrices
    srand(time(NULL));
    randomInit(h_A, m * n);
    randomInit(h_B, n * k);

    // Allocate device memory
    float *d_A, *d_B, *d_C, *d_C_cublas;
    nvtxRangePush("Memory Allocation");
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_A, m * k * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_B, k * n * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_C, m * n * sizeof(float)));
    CUDA_CHECK_ERROR(cudaMalloc((void**)&d_C_cublas, m * n * sizeof(float)));
    nvtxRangePop();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CUDA_CHECK_ERROR(cudaEventCreate(&start));
    CUDA_CHECK_ERROR(cudaEventCreate(&stop));

    nvtxRangePush("Memory copy H2D");
    // Copy matrices to device
    CUDA_CHECK_ERROR(cudaMemcpy(d_A, h_A, m * k * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK_ERROR(cudaMemcpy(d_B, h_B, k * n * sizeof(float), cudaMemcpyHostToDevice));
    nvtxRangePop();

    runCublas(m, n, k, alpha, d_A, d_B, beta, d_C_cublas);
    
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    nvtxRangePush("Memory copy D2H - cuBLAS");
    CUDA_CHECK_ERROR(cudaMemcpy(h_C_cublas, d_C_cublas, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();

    switch(x) {
        case 1:
            printf("Naive GEMM Kernel:\n");
            CUDA_CHECK_ERROR(cudaEventRecord(start));
            runGemmNaive(m, n, k, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK_ERROR(cudaEventRecord(stop));
            CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
            break;
        case 2:
            printf("Global Memory Coalescing:\n");
            CUDA_CHECK_ERROR(cudaEventRecord(start));
            runGemmMemCoalesced(m, n, k, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK_ERROR(cudaEventRecord(stop));
            CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
            break;
        case 3:
            printf("Shared Memory Cache-Blocking:\n");
            CUDA_CHECK_ERROR(cudaEventRecord(start));
            runGemmSharedMem(m, n, k, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK_ERROR(cudaEventRecord(stop));
            CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
            break;
        case 4:
            printf("1D Block tiling:\n");
            CUDA_CHECK_ERROR(cudaEventRecord(start));
            runGemm1dBlockTiling(m, n, k, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK_ERROR(cudaEventRecord(stop));
            CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
            break;
        case 5:
            printf("2D Block tiling:\n");
            CUDA_CHECK_ERROR(cudaEventRecord(start));
            runGemm2dBlockTiling(m, n, k, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK_ERROR(cudaEventRecord(stop));
            CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
            break;
        case 6:
            printf("Vector - 2D Block tiling:\n");
            CUDA_CHECK_ERROR(cudaEventRecord(start));
            runGemmVec2dBlockTiling(m, n, k, alpha, d_A, d_B, beta, d_C);
            CUDA_CHECK_ERROR(cudaEventRecord(stop));
            CUDA_CHECK_ERROR(cudaEventSynchronize(stop));
            break;
        default:
            runGemmNaive(m, n, k, alpha, d_A, d_B, beta, d_C);
            break;
    }

    // Copy result back to host
    nvtxRangePush("Memory copy D2H - Implemented Kernel");
    CUDA_CHECK_ERROR(cudaMemcpy(h_C, d_C, m * n * sizeof(float), cudaMemcpyDeviceToHost));
    nvtxRangePop();

    float cuda_time;
    CUDA_CHECK_ERROR(cudaEventElapsedTime(&cuda_time, start, stop));

    printf("CUDA kernel time: %.4f ms\n", cuda_time);

    bool match = verifyResults(h_C_cublas, h_C, m * n);

    printf("Results match : %s \n",((match)? ("Yes"):("No")));

    // Add this debug printing
    // printf("Initial values after malloc:\n");
    // printf("h_C first 10 elements: ");
    // for(int i = 0; i < 10 && i < m * n; i++) {
    //     printf("%.2f ", h_C[i]);
    // }
    // printf("\n");

    // printf("h_C_cublas first 10 elements: ");
    // for(int i = 0; i < 10 && i < m * n; i++) {
    //     printf("%.2f ", h_C_cublas[i]);
    // }
    // printf("\n\n");

    // Cleanup
    nvtxRangePush("Free memory");
    CUDA_CHECK_ERROR(cudaFree(d_A));
    CUDA_CHECK_ERROR(cudaFree(d_B));
    CUDA_CHECK_ERROR(cudaFree(d_C));
    CUDA_CHECK_ERROR(cudaFree(d_C_cublas));
    nvtxRangePop();
    
    free(h_A);
    free(h_B);
    free(h_C);
    free(h_C_cublas);

    nvtxRangePush("Destroy events");
    CUDA_CHECK_ERROR(cudaEventDestroy(start));
    CUDA_CHECK_ERROR(cudaEventDestroy(stop));
    nvtxRangePop();

    nvtxRangePop();


    return 0;
}