/*
 * bandwidth_sweep.cu
 * Measures memory bandwidth at various transfer sizes.
 * Reveals L2 bandwidth (small sizes) and peak DRAM bandwidth (large sizes).
 * Also used to estimate L2 cache capacity from the bandwidth knee.
 *
 * Output lines: "<bytes> <read_GBs> <write_GBs>"
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

/* Streaming read kernel */
__global__ void stream_read(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    float sum = 0.0f;
    for (; i < n; i += stride) {
        sum += in[i];
    }
    /* Prevent dead-code elimination */
    if (sum == 1.23456789f) out[0] = sum;
}

/* Streaming write kernel */
__global__ void stream_write(float* __restrict__ out, int n, float val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        out[i] = val;
    }
}

/* Copy kernel (read + write) */
__global__ void stream_copy(const float* __restrict__ in, float* __restrict__ out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    for (; i < n; i += stride) {
        out[i] = in[i];
    }
}

double measure_read_bw(const float* d_in, float* d_out, int n, int repeats) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = (blocks > 1024) ? 1024 : blocks;

    /* Warm up */
    stream_read<<<blocks, threads>>>(d_in, d_out, n);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int r = 0; r < repeats; r++) {
        stream_read<<<blocks, threads>>>(d_in, d_out, n);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double bytes = (double)n * sizeof(float) * repeats;
    return bytes / (ms * 1e6); /* GB/s */
}

double measure_write_bw(float* d_out, int n, int repeats) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threads = 256;
    int blocks = (n + threads - 1) / threads;
    blocks = (blocks > 1024) ? 1024 : blocks;

    stream_write<<<blocks, threads>>>(d_out, n, 1.0f);
    cudaDeviceSynchronize();

    cudaEventRecord(start);
    for (int r = 0; r < repeats; r++) {
        stream_write<<<blocks, threads>>>(d_out, n, 1.0f);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms = 0.0f;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    double bytes = (double)n * sizeof(float) * repeats;
    return bytes / (ms * 1e6);
}

int main() {
    /* Test sizes: 256KB to 2GB */
    size_t sizes[] = {
        256  * 1024,
        512  * 1024,
        1    * 1024 * 1024,
        2    * 1024 * 1024,
        4    * 1024 * 1024,
        8    * 1024 * 1024,
        16   * 1024 * 1024,
        32   * 1024 * 1024,
        48   * 1024 * 1024,
        64   * 1024 * 1024,
        128  * 1024 * 1024,
        256  * 1024 * 1024,
        512  * 1024 * 1024,
        1024 * 1024 * 1024,
    };
    int num_sizes = sizeof(sizes) / sizeof(sizes[0]);

    float *d_in = NULL, *d_out = NULL;
    size_t max_size = sizes[num_sizes - 1];

    /* Allocate max size upfront */
    if (cudaMalloc(&d_in, max_size) != cudaSuccess ||
        cudaMalloc(&d_out, max_size) != cudaSuccess) {
        /* Try smaller max */
        max_size = 512 * 1024 * 1024;
        cudaFree(d_in); cudaFree(d_out);
        CHECK(cudaMalloc(&d_in, max_size));
        CHECK(cudaMalloc(&d_out, max_size));
    }

    /* Initialize */
    cudaMemset(d_in, 0, max_size);
    cudaMemset(d_out, 0, max_size);

    for (int si = 0; si < num_sizes; si++) {
        size_t sz = sizes[si];
        if (sz > max_size) break;

        int n = (int)(sz / sizeof(float));
        /* More repeats for small sizes to get stable timing */
        int repeats = (sz < 4 * 1024 * 1024) ? 100 : 10;

        double read_bw = measure_read_bw(d_in, d_out, n, repeats);
        double write_bw = measure_write_bw(d_out, n, repeats);

        printf("%zu %.2f %.2f\n", sz, read_bw, write_bw);
    }

    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
