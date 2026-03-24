## Lab-8 Programs on Matrix using CUDA

```cuda
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
```
Output :
```bash
Enter m,n (Matrix A): 2 2
Enter m,n (Matrix B): 2 2
Enter matrix 1 elements: 3 59
423 4
Enter matrix 2 elements: 4 2 2 4
Enter choice:
1. Row-wise
2. Column-wise
3. Element-wise
1
Resultant Matrix:
7 61 
425 8 
```

```cuda
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
```
Output :
```
Enter m,n (A matrix): 2 3
Enter n,p (B matrix): 3 4
Enter matrix A elements: 2 4 8 1 3 4
Enter matrix B elements: 1 4 8 3 5 2 0 5 3 2 8 59

Enter choice:
1. Row-wise
2. Column-wise
3. Element-wise
2

Resultant Matrix:
46 32 80 498 
28 18 40 254 
```