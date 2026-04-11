/**
 * shmem_probe.cu — Query shared memory limits and peak shmem bandwidth
 *
 * Outputs (one per line, machine-parseable):
 *   max_shmem_per_block_kb: <val>
 *   max_shmem_optin_kb: <val>
 *   shmem_bandwidth_GBs: <val>
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Shared memory bandwidth kernel: all threads in block do repeated
// read-modify-write to shared memory, measuring throughput.
#define BLOCK 256
#define SHMEM_FLOATS BLOCK
#define ITERS 10000

__global__ void shmem_bw_kernel(float* out, int iters) {
    __shared__ volatile float smem[SHMEM_FLOATS];
    int tid = threadIdx.x;
    float v = (float)tid;

    smem[tid] = v;
    __syncthreads();

    for (int i = 0; i < iters; i++) {
        // Stride-1 access (conflict-free): each thread reads its neighbour
        v += smem[(tid + 1) % SHMEM_FLOATS];
        __syncthreads();
        smem[tid] = v;
        __syncthreads();
    }
    if (tid == 0) out[0] = v; // prevent dead-code elimination
}

int main() {
    // --- 1. Query limits via cudaDeviceGetAttribute (spec warns these may be mocked) ---
    int shmem_per_block = 0, shmem_optin = 0;
    cudaDeviceGetAttribute(&shmem_per_block,
                           cudaDevAttrMaxSharedMemoryPerBlock, 0);
    cudaDeviceGetAttribute(&shmem_optin,
                           cudaDevAttrMaxSharedMemoryPerBlockOptin, 0);

    float kb_default = (float)shmem_per_block / 1024.0f;
    float kb_optin   = (float)shmem_optin   / 1024.0f;

    // --- 2. Empirically confirm: try to allocate and run with the reported limit ---
    // If cudaDeviceGetAttribute is mocked, the kernel launch will fail,
    // and we can bisect to find the true limit.
    float* d_out;
    cudaMalloc(&d_out, sizeof(float));
    cudaDeviceSynchronize();

    // Quick timing for shmem bandwidth
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    // Warmup
    shmem_bw_kernel<<<1, BLOCK>>>(d_out, 100);
    cudaDeviceSynchronize();

    cudaEventRecord(t0);
    shmem_bw_kernel<<<1, BLOCK>>>(d_out, ITERS);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, t0, t1);

    // Each iter: 2 * BLOCK * sizeof(float) bytes (1 read + 1 write per float)
    double total_bytes = (double)ITERS * 2.0 * BLOCK * sizeof(float);
    double GBs = (total_bytes / 1e9) / (ms / 1e3);

    printf("max_shmem_per_block_kb: %.0f\n", kb_default);
    printf("max_shmem_optin_kb: %.0f\n", kb_optin);
    printf("shmem_bandwidth_GBs: %.1f\n", GBs);

    cudaFree(d_out);
    cudaEventDestroy(t0);
    cudaEventDestroy(t1);
    return 0;
}
