// Title  : CUDA Program to Multiply Matrices
// Author : Aditya Sinha
// Date   : 20/03/2026

#include<stdio.h>
#include<stdlib.h>
#include<cuda.h>

__global__ void rowMul(const int *a,const int *b,int *c,int m,int n,int p){
	int row = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m){
		for(int j = 0; j < p; j++){
			int sum = 0;
			for(int k = 0; k < n; k++){
				sum += a[row*n + k] * b[k*p + j];
			}
			c[row*p + j] = sum;
		}
	}
}

__global__ void colMul(const int *a,const int *b,int *c,int m,int n,int p){
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(col < p){
		for(int i = 0; i < m; i++){
			int sum = 0;
			for(int k = 0; k < n; k++){
				sum += a[i*n + k] * b[k*p + col];
			}
			c[i*p + col] = sum;
		}
	}
}

__global__ void cellMul(const int *a,const int *b,int *c,int m,int n,int p){
	int row = blockIdx.y * blockDim.y + threadIdx.y;
	int col = blockIdx.x * blockDim.x + threadIdx.x;
	if(row < m && col < p){
		int sum = 0;
		for(int k = 0; k < n; k++){
			sum += a[row*n + k] * b[k*p + col];
		}
		c[row*p + col] = sum;
	}
}

int main(){
	int *a,*b,*c;
	int *d_a,*d_b,*d_c;
	int m,n1,n2,p;
	int i,j;

	printf("Enter m,n (A matrix): ");
	scanf("%d %d",&m,&n1);

	printf("Enter n,p (B matrix): ");
	scanf("%d %d",&n2,&p);

	if(n1 != n2){
		printf("Error: Matrix multiplication not possible!\n");
		return 0;
	}

	int n = n1;

	int sizeA = sizeof(int)*m*n;
	int sizeB = sizeof(int)*n*p;
	int sizeC = sizeof(int)*m*p;

	a = (int*)malloc(sizeA);
	b = (int*)malloc(sizeB);
	c = (int*)malloc(sizeC);

	printf("Enter matrix A elements: ");
	for(i=0;i<m*n;i++){
		scanf("%d",&a[i]);
	}

	printf("Enter matrix B elements: ");
	for(i=0;i<n*p;i++){
		scanf("%d",&b[i]);
	}

	cudaMalloc((void**)&d_a,sizeA);
	cudaMalloc((void**)&d_b,sizeB);
	cudaMalloc((void**)&d_c,sizeC);

	cudaMemcpy(d_a,a,sizeA,cudaMemcpyHostToDevice);
	cudaMemcpy(d_b,b,sizeB,cudaMemcpyHostToDevice);

	int choice;
	printf("\nEnter choice:\n1. Row-wise\n2. Column-wise\n3. Element-wise\n");
	scanf("%d",&choice);

	if(choice == 1){
		int blockSize = 256;
		int gridSize = (m + blockSize - 1)/blockSize;
		rowMul<<<gridSize, blockSize>>>(d_a,d_b,d_c,m,n,p);
	}
	else if(choice == 2){
		int blockSize = 256;
		int gridSize = (p + blockSize - 1)/blockSize;
		colMul<<<gridSize, blockSize>>>(d_a,d_b,d_c,m,n,p);
	}
	else{
		dim3 dimBlock(16,16);
		dim3 dimGrid((p + 15)/16,(m + 15)/16);
		cellMul<<<dimGrid, dimBlock>>>(d_a,d_b,d_c,m,n,p);
	}

	cudaDeviceSynchronize();

	cudaError_t err = cudaGetLastError();
	if(err != cudaSuccess){
		printf("CUDA Error: %s\n",cudaGetErrorString(err));
	}

	cudaMemcpy(c,d_c,sizeC,cudaMemcpyDeviceToHost);

	printf("\nResultant Matrix:\n");
	for(i=0;i<m;i++){
		for(j=0;j<p;j++){
			printf("%d ",c[i*p + j]);
		}
		printf("\n");
	}

	free(a);
	free(b);
	free(c);

	cudaFree(d_a);
	cudaFree(d_b);
	cudaFree(d_c);

	return 0;
}