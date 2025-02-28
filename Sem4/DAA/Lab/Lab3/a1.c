#include <stdio.h>
#include <stdlib.h>

void multiplyMatrices(int **A, int **B, int **C, int r1, int c1, int c2) {
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            C[i][j] = 0;
            for (int k = 0; k < c1; k++) 
                C[i][j] += A[i][k] * B[k][j];
        }
    }
}

int main() {
    int r1, c1, r2, c2;
    
    printf("Enter dimensions of first matrix (rows cols): ");
    scanf("%d %d", &r1, &c1);
    
    printf("Enter dimensions of second matrix (rows cols): ");
    scanf("%d %d", &r2, &c2);

    if (c1 != r2) {
        printf("Matrix multiplication not possible.\n");
        return 0;
    }

    int **A = (int **)malloc(r1 * sizeof(int *));
    int **B = (int **)malloc(r2 * sizeof(int *));
    int **C = (int **)malloc(r1 * sizeof(int *));
    
    for (int i = 0; i < r1; i++) A[i] = (int *)malloc(c1 * sizeof(int));
    for (int i = 0; i < r2; i++) B[i] = (int *)malloc(c2 * sizeof(int));
    for (int i = 0; i < r1; i++) C[i] = (int *)malloc(c2 * sizeof(int));

    printf("Enter elements of first matrix:\n");
    for (int i = 0; i < r1; i++) 
        for (int j = 0; j < c1; j++) 
            scanf("%d", &A[i][j]);

    printf("Enter elements of second matrix:\n");
    for (int i = 0; i < r2; i++) 
        for (int j = 0; j < c2; j++) 
            scanf("%d", &B[i][j]);

    multiplyMatrices(A, B, C, r1, c1, c2);

    printf("Resultant matrix:\n");
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) 
            printf("%d ", C[i][j]);
        printf("\n");
    }

    for (int i = 0; i < r1; i++) free(A[i]);
    for (int i = 0; i < r2; i++) free(B[i]);
    for (int i = 0; i < r1; i++) free(C[i]);
    free(A); free(B); free(C);

    return 0;
}
