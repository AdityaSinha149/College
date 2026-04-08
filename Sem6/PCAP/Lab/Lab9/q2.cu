// Title  : CUDA Program to cahnge rows
// Author : Aditya Sinha
// Date   : 27/03/2026

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void mat_manipulation(int * mat, int * res_mat, int cols) {
    int curr_row = threadIdx.x;

    for (int i = 0; i < cols; i++) {
        int val = mat[curr_row * cols + i];
        for (int j = 0; j < curr_row; j++) {
            val *= mat[curr_row * cols + i];
        }
        res_mat[curr_row * cols + i] = val;
    }
}

int main() {

    int r1, c1;

    printf("Enter dimensions of sparse matrix ");
    scanf("%d %d", &r1, &c1);

    printf("Enter input for matrix 1:\n");
    int * mat = (int *) malloc(sizeof(int) * r1 * c1);
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
            printf("Enter mat[%d][%d] ", i, j);
            scanf("%d", &mat[i * c1 + j]);
        }
    }

    printf("Original Matrix:\n");
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
            printf("%d ", mat[i * c1 + j]);
        }
        printf("\n");
    }

    int * d_mat, *result, *d_result;

    cudaMalloc(&d_mat, sizeof(int) * r1 * c1);
    cudaMalloc(&d_result, sizeof(int) * r1 * c1);

    cudaMemcpy(d_mat, mat, sizeof(int) * r1 * c1, cudaMemcpyHostToDevice);

    mat_manipulation<<<1, r1>>>(d_mat, d_result, c1);

    result = (int *) malloc(sizeof(int) * r1 * c1);
    cudaMemcpy(result, d_result, sizeof(int) * r1 * c1, cudaMemcpyDeviceToHost);

    printf("Resultant Matrix:\n");
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
            printf("%d ", result[i * c1 + j]);
        }
        printf("\n");
    }

    cudaFree(d_mat);
    cudaFree(d_result);
    free(result);

    return 0;
}