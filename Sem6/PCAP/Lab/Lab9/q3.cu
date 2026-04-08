// Title  : CUDA Program to replace non border elems
// Author : Aditya Sinha
// Date   : 27/03/2026

#include <stdio.h>
#include "cuda_runtime.h"

__device__ int onesComplement(int num) {
    int bits = 0, temp = num;

    while (temp > 0) {
        bits++;
        temp >>= 1;
    }

    int mask = (1 << bits) - 1;
    return (~num) & mask;
}

__global__ void processMatrix(int *A, int *B, int M, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < M && j < N) {
        int idx = i * N + j;

        if (i == 0 || j == 0 || i == M - 1 || j == N - 1) {
            B[idx] = A[idx];
        } else {
            B[idx] = onesComplement(A[idx]);
        }
    }
}

int main() {
    int M, N;
    printf("Enter rows and columns: ");
    scanf("%d %d", &M, &N);

    int size = M * N * sizeof(int);

    int *h_A = (int*)malloc(size);
    int *h_B = (int*)malloc(size);

    printf("Enter matrix A:\n");
    for (int i = 0; i < M * N; i++) {
        scanf("%d", &h_A[i]);
    }

    int *d_A, *d_B;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_B, size);

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    dim3 blockSize(16, 16);
    dim3 gridSize((N + 15) / 16, (M + 15) / 16);

    processMatrix<<<gridSize, blockSize>>>(d_A, d_B, M, N);

    cudaMemcpy(h_B, d_B, size, cudaMemcpyDeviceToHost);

    printf("\nOutput Matrix B:\n");
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            printf("%d ", h_B[i * N + j]);
        }
        printf("\n");
    }

    cudaFree(d_A);
    cudaFree(d_B);
    free(h_A);
    free(h_B);

    return 0;
}