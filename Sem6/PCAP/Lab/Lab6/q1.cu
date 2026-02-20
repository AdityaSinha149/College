// Title  : 1D Convolution using CUDA
// Author : Aditya Sinha
// Date   : 20/02/2026

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void convolution1D(const int *input, const int *mask, int *output, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        int radius = m / 2;
        int sum = 0;
        for (int j = 0; j < m; ++j) {
            int inputIndex = idx + j - radius;
            if (inputIndex >= 0 && inputIndex < n) {
                sum += input[inputIndex] * mask[j];
            }
        }
        output[idx] = sum;
    }
}

static void convolution1DHost(const int *input, const int *mask, int *output, int n, int m) {
    int radius = m / 2;
    for (int idx = 0; idx < n; ++idx) {
        int sum = 0;
        for (int j = 0; j < m; ++j) {
            int inputIndex = idx + j - radius;
            if (inputIndex >= 0 && inputIndex < n) {
                sum += input[inputIndex] * mask[j];
            }
        }
        output[idx] = sum;
    }
}

static int convolution1DCuda(const int *h_input, const int *h_mask, int *h_output, int n, int m) {
    int inputBytes = n * (int)sizeof(int);
    int maskBytes = m * (int)sizeof(int);

    int *d_input = NULL;
    int *d_mask = NULL;
    int *d_output = NULL;

    cudaError_t err = cudaMalloc((void **)&d_input, inputBytes);
    if (err != cudaSuccess) {
        return 1;
    }
    err = cudaMalloc((void **)&d_mask, maskBytes);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        return 1;
    }
    err = cudaMalloc((void **)&d_output, inputBytes);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_mask);
        return 1;
    }

    err = cudaMemcpy(d_input, h_input, inputBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_mask);
        cudaFree(d_output);
        return 1;
    }
    err = cudaMemcpy(d_mask, h_mask, maskBytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_mask);
        cudaFree(d_output);
        return 1;
    }

    int threads = 256;
    int blocks = (n + threads - 1) / threads;

    convolution1D<<<blocks, threads>>>(d_input, d_mask, d_output, n, m);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_mask);
        cudaFree(d_output);
        return 1;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_mask);
        cudaFree(d_output);
        return 1;
    }

    err = cudaMemcpy(h_output, d_output, inputBytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_input);
        cudaFree(d_mask);
        cudaFree(d_output);
        return 1;
    }

    cudaFree(d_input);
    cudaFree(d_mask);
    cudaFree(d_output);
    return 0;
}

int main() {
    int n = 0;
    int m = 0;
    scanf("%d", &n);
    scanf("%d", &m);

    if (n <= 0 || m <= 0) {
        return 1;
    }

    int inputBytes = n * (int)sizeof(int);
    int maskBytes = m * (int)sizeof(int);

    int *h_input = (int *)malloc(inputBytes);
    int *h_mask = (int *)malloc(maskBytes);
    int *h_output = (int *)malloc(inputBytes);

    if (h_input == NULL || h_mask == NULL || h_output == NULL) {
        free(h_input);
        free(h_mask);
        free(h_output);
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        scanf("%d", &h_input[i]);
    }
    for (int i = 0; i < m; ++i) {
        scanf("%d", &h_mask[i]);
    }

    if (convolution1DCuda(h_input, h_mask, h_output, n, m) != 0) {
        convolution1DHost(h_input, h_mask, h_output, n, m);
    }

    for (int i = 0; i < n; ++i) {
        printf("%d", h_output[i]);
        if (i + 1 < n) {
            printf(" ");
        }
    }
    printf("\n");

    free(h_input);
    free(h_mask);
    free(h_output);

    return 0;
}
