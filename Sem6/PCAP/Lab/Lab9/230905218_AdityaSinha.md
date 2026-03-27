## Lab-9 Programs on Matrix using CUDA 2

```cuda
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
```
Output :
```bash
Enter dimensions of sparse matrix 3 3
Enter input for matrix 1:
Enter mat[0][0] 1
Enter mat[0][1] 0
Enter mat[0][2] 2
Enter mat[1][0] 0
Enter mat[1][1] 3
Enter mat[1][2] 0
Enter mat[2][0] 4
Enter mat[2][1] 0
Enter mat[2][2] 5
Enter input for vector:
Enter vector[0] 1
Enter vector[1] 2
Enter vector[2] 3

Sparse Matrix:
1 0 2 
0 3 0 
4 0 5 

Vector:
1 2 3 

CSR format:
Row pointers: 0 2 3 5 
Column offsets: 0 2 1 0 2 
Data: 1 2 3 4 5 

Result:
7 6 19 
```

```cuda
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
```
Output :
```
Enter dimensions of sparse matrix 3 3
Enter input for matrix 1:
Enter mat[0][0] 1
Enter mat[0][1] 2
Enter mat[0][2] 3
Enter mat[1][0] 4
Enter mat[1][1] 5
Enter mat[1][2] 6
Enter mat[2][0] 7
Enter mat[2][1] 8
Enter mat[2][2] 9

Original Matrix:
1 2 3 
4 5 6 
7 8 9 

Resultant Matrix:
1 2 3 
16 25 36 
343 512 729 
```

```cuda
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
```
Output :
```
Enter rows and columns: 4 4
Enter matrix A:
1 2 3 4
6 5 8 3
2 4 10 1
9 1 2 5

Output Matrix B:
1 2 3 4 
6 2 7 3 
2 3 5 1 
9 1 2 5 
```