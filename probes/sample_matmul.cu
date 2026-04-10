/*
 * sample_matmul.cu — simple matrix multiply for testing operator profiling
 */
#include <stdio.h>
#include <cuda_runtime.h>

#define N 1024

__global__ void matmul(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= n || col >= n) return;
    float sum = 0.0f;
    for (int k = 0; k < n; k++) {
        sum += A[row * n + k] * B[k * n + col];
    }
    C[row * n + col] = sum;
}

int main() {
    size_t sz = N * N * sizeof(float);
    float *dA, *dB, *dC;
    cudaMalloc(&dA, sz); cudaMalloc(&dB, sz); cudaMalloc(&dC, sz);
    cudaMemset(dA, 1, sz); cudaMemset(dB, 1, sz);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    // Warm up
    matmul<<<grid, block>>>(dA, dB, dC, N);
    cudaDeviceSynchronize();

    // Actual run
    for (int i = 0; i < 5; i++) {
        matmul<<<grid, block>>>(dA, dB, dC, N);
    }
    cudaDeviceSynchronize();

    printf("matmul done\n");
    cudaFree(dA); cudaFree(dB); cudaFree(dC);
    return 0;
}
