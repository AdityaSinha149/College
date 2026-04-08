// Title  : parrallel mat multipliucation
// Author : Aditya Sinha
// Date   : 27/03/2026

#include "cuda_runtime.h"
#include <stdio.h>
#include <stdlib.h>

__global__ void spmv(int * row_ptrs, int * col_offsets, int * data, int * vector, int * result, int vector_size) {
    int i = threadIdx.x;
    int start = row_ptrs[i], end = row_ptrs[i + 1];
    int sum = 0;
    for (int j = start; j < end; j++) {
        sum += data[j] * vector[col_offsets[j]];
    }
    result[i] = sum;
}

void to_csr_format(int * mat, int r, int c, int ** row_ptrs, int ** col_offsets, int ** data, int * ret_row_ptrs_count, int * ret_data_count) {
    int row_ptrs_count = 0, data_count = 0;
    for (int i = 0; i < r; i++) {
        int inserted_row_ptr = 0;
        for (int j = 0; j < c; j++) {
            int ele_ind = i * c + j;
            if (mat[ele_ind] != 0) {
                if (inserted_row_ptr == 0) {
                    row_ptrs_count++;
                    *row_ptrs = (int *) realloc(*row_ptrs, sizeof(int) * row_ptrs_count);
                    (*row_ptrs)[row_ptrs_count - 1] = data_count;
                    inserted_row_ptr = 1;
                }
                data_count++;
                *data = (int *) realloc(*data, sizeof(int) * data_count);
                *col_offsets = (int *) realloc(*col_offsets, sizeof(int) * data_count);
                (*data)[data_count - 1] = mat[ele_ind];
                (*col_offsets)[data_count - 1] = j;
            }
        }
        if (inserted_row_ptr == 0) {
            row_ptrs_count++;
            *row_ptrs = (int *) realloc(*row_ptrs, sizeof(int) * row_ptrs_count);
            (*row_ptrs)[row_ptrs_count - 1] = data_count;
        }
    }
    row_ptrs_count++;
    *row_ptrs = (int *) realloc(*row_ptrs, sizeof(int) * row_ptrs_count);
    (*row_ptrs)[row_ptrs_count - 1] = data_count;
    *ret_row_ptrs_count = row_ptrs_count;
    *ret_data_count = data_count;
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

    printf("Enter input for vector:\n");
    int * vector = (int *) malloc(sizeof(int) * c1);
    for (int i = 0; i < c1; i++) {
        printf("Enter vector[%d] ", i);
        scanf("%d", &vector[i]);
    }

    printf("Sparse Matrix:\n");
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c1; j++) {
            printf("%d ", mat[i * c1 + j]);
        }
        printf("\n");
    }

    printf("Vector:\n");
    for (int i = 0; i < c1; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");


    int * mat_row_ptrs = (int *) malloc(sizeof(int)), mat_row_ptrs_count = 0;
    int * mat_col_offsets = (int *) malloc(sizeof(int));
    int * mat_data = (int *) malloc(sizeof(int)), mat_data_count = 0;

    to_csr_format(mat, r1, c1, &mat_row_ptrs, &mat_col_offsets, &mat_data, &mat_row_ptrs_count, &mat_data_count);

    printf("CSR format:\n");
    printf("Row pointers: ");
    for (int i = 0; i < mat_row_ptrs_count; i++) {
        printf("%d ", mat_row_ptrs[i]);
    }
    printf("\n");
    printf("Column offsets: ");
    for (int i = 0; i < mat_data_count; i++) {
        printf("%d ", mat_col_offsets[i]);
    }
    printf("\n");
    printf("Data: ");
    for (int i = 0; i < mat_data_count; i++) {
        printf("%d ", mat_data[i]);
    }
    printf("\n");

    int * d_vector, * d_result, * d_row_ptrs, * d_col_offsets, * d_data, * result;

    cudaMalloc(&d_vector, sizeof(int) * c1);
    cudaMalloc(&d_result, sizeof(int) * r1);
    cudaMalloc(&d_row_ptrs, sizeof(int) * mat_row_ptrs_count);
    cudaMalloc(&d_col_offsets, sizeof(int) * mat_data_count);
    cudaMalloc(&d_data, sizeof(int) * mat_data_count);

    cudaMemcpy(d_vector, vector, sizeof(int) * c1, cudaMemcpyHostToDevice);
    cudaMemcpy(d_row_ptrs, mat_row_ptrs, sizeof(int) * mat_row_ptrs_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_offsets, mat_col_offsets, sizeof(int) * mat_data_count, cudaMemcpyHostToDevice);
    cudaMemcpy(d_data, mat_data, sizeof(int) * mat_data_count, cudaMemcpyHostToDevice);

    spmv<<<1, r1>>>(d_row_ptrs, d_col_offsets, d_data, d_vector, d_result, c1);

    result = (int *) malloc(sizeof(int) * r1);
    cudaMemcpy(result, d_result, sizeof(int) * r1, cudaMemcpyDeviceToHost);

    printf("Result:\n");
    for (int i = 0; i < r1; i++) {
        printf("%d ", result[i]);
    }
    printf("\n");

    free(mat_row_ptrs);
    free(mat_col_offsets);
    free(mat_data);
    free(result);

    cudaFree(d_vector);
    cudaFree(d_result);
    cudaFree(d_row_ptrs);
    cudaFree(d_col_offsets);
    cudaFree(d_data);

    return 0;
}