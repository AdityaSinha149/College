// Title  : CUDA Program to Add Two Matrices
// Author : Aditya Sinha
// Date   : 20/03/2026

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

__global__ void rowAdd(int *a,int *b,int *t,int m,int n){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m){
		for(int i = 0; i < n; i++){
			t[row*n + i] = a[row*n + i] + b[row*n + i];
		}
	}
}

__global__ void colAdd(int *a,int *b,int *t,int m,int n){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < n){
		for(int i = 0; i < m; i++){
			t[i*n + col] = a[i*n + col] + b[i*n + col];
		}
	}
}

__global__ void cellAdd(int *a,int *b,int *t,int m,int n){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m && col < n){
		int idx = row*n + col;
		t[idx] = a[idx] + b[idx];
	}
}

int main(){
	int *a,*b,*t;
	int *d_a,*d_b,*d_t;
	int m1,n1,m2,n2,i,j;

	printf("Enter m,n (Matrix A): ");
	scanf("%d %d",&m1,&n1);

	printf("Enter m,n (Matrix B): ");
	scanf("%d %d",&m2,&n2);

	if(m1 != m2 || n1 != n2){
		printf("Error: Matrix addition not possible!\n");
		return 0;
	}

	int m = m1;
	int n = n1;

	int size = sizeof(int)*m*n;

	a = (int*)malloc(size);
	b = (int*)malloc(size);
	t = (int*)malloc(size);

	printf("Enter matrix 1 elements: ");
	for(i = 0; i < m*n; i++){
		scanf("%d",&a[i]);
	}

	printf("Enter matrix 2 elements: ");
	for(i = 0; i < m*n; i++){
		scanf("%d",&b[i]);
	}

	cudaMalloc((void**)&d_a,size);
	cudaMalloc((void**)&d_b,size);
	cudaMalloc((void**)&d_t,size);

	cudaMemcpy(d_a,a,size,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,size,cudaMemcpyHostToDevice);

	int choice;
	printf("Enter choice:\n1. Row-wise\n2. Column-wise\n3. Element-wise\n");
	scanf("%d",&choice);

	if(choice == 1){
		int blockSize = 256;
		int gridSize = (m + blockSize - 1)/blockSize;
		rowAdd<<<gridSize, blockSize>>>(d_a,d_b,d_t,m,n);
	}
	else if(choice == 2){
		int blockSize = 256;
		int gridSize = (n + blockSize - 1)/blockSize;
		colAdd<<<gridSize, blockSize>>>(d_a,d_b,d_t,m,n);
	}
	else{
		dim3 dimBlock(16,16);
		dim3 dimGrid((n + 15)/16,(m + 15)/16);
		cellAdd<<<dimGrid, dimBlock>>>(d_a,d_b,d_t,m,n);
	}

	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("CUDA error: %s\n",cudaGetErrorString(err));
	}

	cudaMemcpy(t,d_t,size,cudaMemcpyDeviceToHost);

	printf("Resultant Matrix:\n");
	for(i = 0; i < m; i++){
		for(j = 0; j < n; j++){
			printf("%d ",t[i*n + j]);
		}
		printf("\n");
	}

	free(a);
	free(b);
	free(t);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_t);

	return 0;
}