// Title  : Radians to Sine using CUDA
// Author : Aditya Sinha
// Date   : 13/02/2026

#include <stdio.h>
#include <cuda_runtime.h>
#include <stdlib.h>

__global__ void radiansToSine(float *angles, float *sines, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        const float angle = angles[idx];
        sines[idx] = sinf(angle);
    }
    return;
}

int main() {
    int n;
    scanf("%d", &n);

    float *h_angles = (float *)malloc(n * sizeof(float));
    float *h_sines = (float *)malloc(n * sizeof(float));
    for (int i = 0; i < n; ++i) {
        scanf("%f", &h_angles[i]);
    }

    float *d_angles;
    float *d_sines;
    cudaMalloc((void **)&d_angles, n * sizeof(float));
    cudaMalloc((void **)&d_sines, n * sizeof(float));
    cudaMemcpy(d_angles, h_angles, n * sizeof(float), cudaMemcpyHostToDevice);

    radiansToSine<<<(n + 255) / 256, 256>>>(d_angles, d_sines, n);
    cudaMemcpy(h_sines, d_sines, n * sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        printf("%f ", h_sines[i]);
    }
    printf("\n");
    cudaFree(d_angles);
    cudaFree(d_sines);
    free(h_angles);
    free(h_sines);

    return 0;
}