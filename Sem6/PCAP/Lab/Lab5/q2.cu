// Title  : Vector Addition using CUDA
// Author : Aditya Sinha
// Date   : 13/02/2026

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static const int threadsPerBlock = 256;

__global__ void vectorAdd(const int *a, const int *b, int *c, int n) {
	const int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx < n) {
		c[idx] = a[idx] + b[idx];
	}
}

int main() {
	int lenA = 0;
	int lenB = 0;
	scanf("%d", &lenA);
	scanf("%d", &lenB);

	if (lenA != lenB) {
		printf("Vector size mismatch: %d vs %d\n", lenA, lenB);
		return 1;
	}

	const int n = lenA;
	const int bytes = n * (int)sizeof(int);

	int *h_a = (int *)malloc(bytes);
	int *h_b = (int *)malloc(bytes);
	int *h_c = (int *)malloc(bytes);

	for (int i = 0; i < n; ++i) {
		h_a[i] = i;
		h_b[i] = 2*i;
	}

	int *d_a = NULL;
	int *d_b = NULL;
	int *d_c = NULL;
	cudaMalloc((void **)&d_a, bytes);
	cudaMalloc((void **)&d_b, bytes);
	cudaMalloc((void **)&d_c, bytes);

	cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);

	const int blocks = (n + threadsPerBlock - 1) / threadsPerBlock;

	vectorAdd<<<blocks, threadsPerBlock>>>(d_a, d_b, d_c, n);

	cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);

	printf("Vector length: %d\n", n);
	printf("Threads per block: %d\n", threadsPerBlock);
	printf("Blocks launched: %d\n", blocks);
    for(int i = 0; i < n; ++i) {
        printf("%d ", h_c[i]);
    }
    printf("\n");

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);
	free(h_a);
	free(h_b);
	free(h_c);

	return 0;
}
