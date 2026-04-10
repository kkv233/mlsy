/*
 * clock_probe.cu
 * Measures actual SM boost clock frequency by correlating clock64() ticks
 * with wall-clock time via CUDA events.
 *
 * Output: "clock_mhz: <value>"
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

/* Warm-up kernel: keep SM busy to reach boost clock */
__global__ void warmup(volatile float* data, int n, int iters) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? data[idx] : 0.0f;
    for (int i = 0; i < iters; i++) {
        val = val * 1.0001f + 0.0001f;
    }
    if (idx < n) data[idx] = val;
}

/* Clock measurement kernel: single thread, single block */
__global__ void measure_clock(uint64_t* ticks_out, int iters) {
    /* Prevent compiler from optimizing away the loop */
    volatile int x = threadIdx.x;
    uint64_t t0 = clock64();
    for (int i = 0; i < iters; i++) {
        x = x * 3 + 1;
    }
    uint64_t t1 = clock64();
    ticks_out[0] = t1 - t0;
    ticks_out[1] = (uint64_t)x; /* prevent dead-code elimination */
}

int main() {
    const int WARMUP_ITERS = 10000;
    const int MEASURE_ITERS = 100000000; /* 100M iterations */

    /* Warm up GPU to reach boost clock */
    float* d_warmup;
    CHECK(cudaMalloc(&d_warmup, 1024 * sizeof(float)));
    warmup<<<32, 256>>>(d_warmup, 1024, WARMUP_ITERS);
    CHECK(cudaDeviceSynchronize());
    cudaFree(d_warmup);

    /* Measure clock */
    uint64_t* d_ticks;
    uint64_t h_ticks[2] = {0, 0};
    CHECK(cudaMalloc(&d_ticks, 2 * sizeof(uint64_t)));

    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    CHECK(cudaEventRecord(start));
    measure_clock<<<1, 1>>>(d_ticks, MEASURE_ITERS);
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));

    float elapsed_ms = 0.0f;
    CHECK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    CHECK(cudaMemcpy(h_ticks, d_ticks, 2 * sizeof(uint64_t), cudaMemcpyDeviceToHost));

    double clock_mhz = (double)h_ticks[0] / (elapsed_ms * 1000.0);

    printf("clock_mhz: %.2f\n", clock_mhz);
    fprintf(stderr, "ticks=%llu elapsed_ms=%.3f\n",
            (unsigned long long)h_ticks[0], elapsed_ms);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_ticks);
    return 0;
}
