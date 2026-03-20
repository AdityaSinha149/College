// Title  : CUDA program to generate RS from string S
// Author : Aditya Sinha
// Date   : 13/03/2026

#include <stdio.h>
#include <string.h>
#include <cuda.h>

__global__ void generateRS(char *S, char *RS, int n)
{
    int i = threadIdx.x;

    int pos = 0;

    for (int k = n; k > 0; k--)
    {
        if (i < k)
            RS[pos + i] = S[i];

        pos += k;
    }
}

int main()
{
    char S[100];
    char *d_S, *d_RS;

    printf("Enter string: ");
    scanf("%s", S);

    int n = strlen(S);
    int size = n * (n + 1) / 2;

    char RS[size + 1];

    cudaMalloc(&d_S, n);
    cudaMalloc(&d_RS, size);

    cudaMemcpy(d_S, S, n, cudaMemcpyHostToDevice);

    generateRS<<<1, n>>>(d_S, d_RS, n);

    cudaMemcpy(RS, d_RS, size, cudaMemcpyDeviceToHost);
    RS[size] = '\0';

    printf("RS = %s\n", RS);

    cudaFree(d_S);
    cudaFree(d_RS);

    return 0;
}