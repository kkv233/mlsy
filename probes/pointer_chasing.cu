/*
 * pointer_chasing.cu
 * Measures memory latency hierarchy (L1/L2/DRAM) via random pointer chasing.
 * Defeats hardware prefetcher by using a random permutation linked list.
 *
 * Output lines: "<size_bytes> <latency_cycles>"
 */
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

/* Single-thread pointer chasing kernel */
__global__ void chase_kernel(uint32_t* arr, uint64_t* out, int chain_len) {
    /* Single thread only — no parallelism to avoid cache sharing effects */
    if (threadIdx.x != 0 || blockIdx.x != 0) return;

    /* Force load from global memory, prevent optimization */
    uint32_t idx = arr[0];
    uint64_t t0 = clock64();
    for (int i = 0; i < chain_len; i++) {
        idx = arr[idx];
    }
    uint64_t t1 = clock64();

    out[0] = t1 - t0;
    out[1] = idx; /* prevent dead-code elimination */
}

/* Fisher-Yates shuffle to create random permutation */
void make_random_permutation(uint32_t* arr, int n) {
    for (int i = 0; i < n; i++) arr[i] = i;
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        uint32_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
    }
    /* Ensure no self-loops: arr[i] != i */
    for (int i = 0; i < n; i++) {
        if (arr[i] == (uint32_t)i) {
            int j = (i + 1) % n;
            uint32_t tmp = arr[i]; arr[i] = arr[j]; arr[j] = tmp;
        }
    }
}

int main() {
    srand(42);

    /* Array sizes to test: from 32KB to 256MB */
    size_t sizes[] = {
        32   * 1024,          /*  32 KB  — fits in L1 (128KB) */
        64   * 1024,          /*  64 KB */
        128  * 1024,          /* 128 KB  — L1 boundary */
        256  * 1024,          /* 256 KB  — in L2 */
        512  * 1024,          /* 512 KB */
        1    * 1024 * 1024,   /*   1 MB */
        2    * 1024 * 1024,   /*   2 MB */
        4    * 1024 * 1024,   /*   4 MB */
        8    * 1024 * 1024,   /*   8 MB */
        16   * 1024 * 1024,   /*  16 MB */
        32   * 1024 * 1024,   /*  32 MB */
        48   * 1024 * 1024,   /*  48 MB  — near L2 boundary (~50MB) */
        64   * 1024 * 1024,   /*  64 MB  — beyond L2 */
        128  * 1024 * 1024,   /* 128 MB  — DRAM */
        256  * 1024 * 1024,   /* 256 MB  — DRAM */
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);
    const int CHAIN_LEN = 4096; /* enough to amortize kernel launch overhead */

    uint64_t* d_out;
    CHECK(cudaMalloc(&d_out, 2 * sizeof(uint64_t)));

    for (int si = 0; si < num_sizes; si++) {
        size_t sz = sizes[si];
        int n = (int)(sz / sizeof(uint32_t));
        if (n < CHAIN_LEN) n = CHAIN_LEN;

        /* Build random permutation on host */
        uint32_t* h_arr = (uint32_t*)malloc(n * sizeof(uint32_t));
        if (!h_arr) { fprintf(stderr, "malloc failed for size %zu\n", sz); continue; }
        make_random_permutation(h_arr, n);

        uint32_t* d_arr;
        if (cudaMalloc(&d_arr, n * sizeof(uint32_t)) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed for size %zu\n", sz);
            free(h_arr);
            continue;
        }
        CHECK(cudaMemcpy(d_arr, h_arr, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
        free(h_arr);

        /* Warm up */
        uint64_t h_out[2];
        chase_kernel<<<1, 1>>>(d_arr, d_out, CHAIN_LEN / 4);
        CHECK(cudaDeviceSynchronize());

        /* Measure (3 runs, take median) */
        uint64_t measurements[3];
        for (int r = 0; r < 3; r++) {
            CHECK(cudaMemset(d_out, 0, 2 * sizeof(uint64_t)));
            chase_kernel<<<1, 1>>>(d_arr, d_out, CHAIN_LEN);
            CHECK(cudaDeviceSynchronize());
            CHECK(cudaMemcpy(h_out, d_out, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));
            measurements[r] = h_out[0];
        }
        /* Simple median of 3 */
        uint64_t med;
        if (measurements[0] > measurements[1]) { uint64_t t = measurements[0]; measurements[0] = measurements[1]; measurements[1] = t; }
        if (measurements[1] > measurements[2]) { uint64_t t = measurements[1]; measurements[1] = measurements[2]; measurements[2] = t; }
        if (measurements[0] > measurements[1]) { uint64_t t = measurements[0]; measurements[0] = measurements[1]; measurements[1] = t; }
        med = measurements[1];

        double latency_per_access = (double)med / CHAIN_LEN;
        printf("%zu %.2f\n", sz, latency_per_access);

        cudaFree(d_arr);
    }

    cudaFree(d_out);
    return 0;
}
