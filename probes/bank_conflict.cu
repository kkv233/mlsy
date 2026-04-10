/*
 * bank_conflict.cu
 * Measures shared memory bank conflict penalty by varying access stride.
 * stride=1: no conflict (consecutive threads access consecutive banks)
 * stride=32: 32-way conflict (all threads hit same bank)
 *
 * Output lines: "<stride> <cycles_per_access>"
 */
#include <stdio.h>
#include <stdint.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

#define SMEM_BANKS  32
#define SMEM_SIZE   (SMEM_BANKS * 32)  /* 1024 ints = 4KB */
#define WARP_SIZE   32
#define ITERS       10000

/*
 * Bank conflict measurement using a single warp.
 * stride=1: thread i accesses bank i (no conflict)
 * stride=32: all threads access bank 0 (32-way conflict)
 *
 * We use a read-modify-write pattern to prevent the compiler from
 * eliminating the shared memory accesses.
 */
__global__ void smem_stride_kernel(int stride, uint64_t* out) {
    __shared__ volatile int smem[SMEM_SIZE];

    int tid = threadIdx.x;

    /* Initialize — all threads participate */
    for (int i = tid; i < SMEM_SIZE; i += blockDim.x) {
        smem[i] = i + 1;
    }
    __syncthreads();

    if (tid >= WARP_SIZE) return;

    /* Each thread's base index: stride apart, wrapping within SMEM_SIZE */
    int base = (tid * stride) & (SMEM_SIZE - 1);

    int acc = 0;
    uint64_t t0 = clock64();

    #pragma unroll 1
    for (int iter = 0; iter < ITERS; iter++) {
        /* Read-modify-write to force actual shared memory traffic */
        int val = smem[base];
        smem[base] = val + 1;
        acc += val;
        /* Rotate base to prevent compiler from predicting address */
        base = (base + stride) & (SMEM_SIZE - 1);
    }

    uint64_t t1 = clock64();

    if (tid == 0) {
        out[0] = t1 - t0;
        out[1] = (uint64_t)acc;
    }
}

int main() {
    uint64_t* d_out;
    CHECK(cudaMalloc(&d_out, 2 * sizeof(uint64_t)));

    uint64_t h_out[2];

    /* Test strides 1..32 */
    for (int stride = 1; stride <= 32; stride++) {
        /* 3 runs, take median */
        uint64_t measurements[3];
        for (int r = 0; r < 3; r++) {
            CHECK(cudaMemset(d_out, 0, 2 * sizeof(uint64_t)));
            smem_stride_kernel<<<1, 64>>>(stride, d_out);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out, d_out, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            measurements[r] = h_out[0];
        }
        /* Sort and take median */
        for (int i = 0; i < 2; i++)
            for (int j = i+1; j < 3; j++)
                if (measurements[i] > measurements[j]) {
                    uint64_t t = measurements[i]; measurements[i] = measurements[j]; measurements[j] = t;
                }
        uint64_t med = measurements[1];
        double cycles_per_access = (double)med / ITERS;
        printf("%d %.4f\n", stride, cycles_per_access);
    }

    cudaFree(d_out);
    return 0;
}
