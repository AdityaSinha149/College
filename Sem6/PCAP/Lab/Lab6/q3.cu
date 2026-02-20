// Title  : Odd-Even Transposition Sort using CUDA
// Author : Aditya Sinha
// Date   : 20/02/2026

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void oddEvenPhase(int *arr, int n, int phase) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int i = 2 * tid + phase;
    if (i + 1 < n) {
        int left = arr[i];
        int right = arr[i + 1];
        if (left > right) {
            arr[i] = right;
            arr[i + 1] = left;
        }
    }
}

static void oddEvenSortHost(int *arr, int n) {
    for (int iter = 0; iter < n; ++iter) {
        int start = iter % 2;
        for (int i = start; i + 1 < n; i += 2) {
            if (arr[i] > arr[i + 1]) {
                int temp = arr[i];
                arr[i] = arr[i + 1];
                arr[i + 1] = temp;
            }
        }
    }
}

static int oddEvenSortCuda(int *h_arr, int n) {
    int bytes = n * (int)sizeof(int);
    int *d_arr = NULL;
    cudaError_t err = cudaMalloc((void **)&d_arr, bytes);
    if (err != cudaSuccess) {
        return 1;
    }
    err = cudaMemcpy(d_arr, h_arr, bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cudaFree(d_arr);
        return 1;
    }

    int threads = 256;
    int pairs = n / 2;
    int blocks = (pairs + threads - 1) / threads;

    for (int iter = 0; iter < n; ++iter) {
        oddEvenPhase<<<blocks, threads>>>(d_arr, n, 0);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_arr);
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_arr);
            return 1;
        }
        oddEvenPhase<<<blocks, threads>>>(d_arr, n, 1);
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            cudaFree(d_arr);
            return 1;
        }
        err = cudaDeviceSynchronize();
        if (err != cudaSuccess) {
            cudaFree(d_arr);
            return 1;
        }
    }

    err = cudaMemcpy(h_arr, d_arr, bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cudaFree(d_arr);
        return 1;
    }

    cudaFree(d_arr);
    return 0;
}

int main() {
    int n = 0;
    scanf("%d", &n);
    if (n <= 0) {
        return 1;
    }

    int bytes = n * (int)sizeof(int);
    int *h_arr = (int *)malloc(bytes);
    if (h_arr == NULL) {
        return 1;
    }

    for (int i = 0; i < n; ++i) {
        scanf("%d", &h_arr[i]);
    }

    if (oddEvenSortCuda(h_arr, n) != 0) {
        oddEvenSortHost(h_arr, n);
    }

    for (int i = 0; i < n; ++i) {
        printf("%d", h_arr[i]);
        if (i + 1 < n) {
            printf(" ");
        }
    }
    printf("\n");

    free(h_arr);
    return 0;
}
