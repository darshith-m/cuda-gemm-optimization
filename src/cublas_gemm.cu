#include <cuda_runtime.h>
#include <cublas_v2.h>

void runCublas(int m, int n, int k, float alpha, float* d_A, float* d_B, float beta, float* d_C) {
   cublasHandle_t handle;
   cublasCreate(&handle);
   
   cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
               n, m, k,  // Corrected order: n,m,k for cuBLAS
               &alpha,
               d_B, n,   // Leading dimension is n
               d_A, k,   // Leading dimension is k
               &beta,
               d_C, n);  // Leading dimension is n
               
   cublasDestroy(handle);
}