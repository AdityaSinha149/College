#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int **a, **b, **c;
int r1, c1, r2, c2;

void *multiplyElement(void *arg) {
    int idx = *(int*)arg;
    free(arg);
    int row = idx / c2;
    int col = idx % c2;

    c[row][col] = 0;
    for (int k = 0; k < c1; k++)
        c[row][col] += a[row][k] * b[k][col];

    return NULL;
}

int main() {
    printf("Enter rows and cols of matrix A: ");
    scanf("%d %d", &r1, &c1);
    printf("Enter rows and cols of matrix B: ");
    scanf("%d %d", &r2, &c2);

    if (c1 != r2) {
        printf("Matrix multiplication not possible.\n");
        return 0;
    }

    // Allocate matrices
    a = (int**)malloc(r1 * sizeof(int*));
    b = (int**)malloc(r2 * sizeof(int*));
    c = (int**)malloc(r1 * sizeof(int*));
    for (int i = 0; i < r1; i++) {
        a[i] = (int*)malloc(c1 * sizeof(int));
        c[i] = (int*)malloc(c2 * sizeof(int));
    }
    for (int i = 0; i < r2; i++)
        b[i] = (int*)malloc(c2 * sizeof(int));

    printf("Enter elements of A:\n");
    for (int i = 0; i < r1; i++)
        for (int j = 0; j < c1; j++)
            scanf("%d", &a[i][j]);

    printf("Enter elements of B:\n");
    for (int i = 0; i < r2; i++)
        for (int j = 0; j < c2; j++)
            scanf("%d", &b[i][j]);

    pthread_t *threads = (pthread_t*)malloc(r1 * c2 * sizeof(pthread_t));

    int tCount = 0;
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++) {
            int *idx = (int*)malloc(sizeof(int));
            *idx = i * c2 + j;
            pthread_create(&threads[tCount++], NULL, multiplyElement, idx);
        }
    }

    for (int i = 0; i < tCount; i++)
        pthread_join(threads[i], NULL);

    printf("Resultant Matrix:\n");
    for (int i = 0; i < r1; i++) {
        for (int j = 0; j < c2; j++)
            printf("%d ", c[i][j]);
        printf("\n");
    }

    // Free memory
    for (int i = 0; i < r1; i++) { free(a[i]); free(c[i]); }
    for (int i = 0; i < r2; i++) free(b[i]);
    free(a); free(b); free(c); free(threads);

    return 0;
}
