#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int **mat;
int *rowSum, *colSum;
int r, c;

void *calcSum(void *arg) {
    int *data = (int*)arg;
    int idx = data[0];
    int type = data[1]; // 0 = row, 1 = column
    free(arg);

    if (type == 0) { // row sum
        rowSum[idx] = 0;
        for (int j = 0; j < c; j++)
            rowSum[idx] += mat[idx][j];
    } else { // column sum
        colSum[idx] = 0;
        for (int i = 0; i < r; i++)
            colSum[idx] += mat[i][idx];
    }
    return NULL;
}

int main() {
    printf("Enter rows and cols of matrix: ");
    scanf("%d %d", &r, &c);

    mat = (int**)malloc(r * sizeof(int*));
    for (int i = 0; i < r; i++)
        mat[i] = (int*)malloc(c * sizeof(int));

    rowSum = (int*)malloc(r * sizeof(int));
    colSum = (int*)malloc(c * sizeof(int));

    printf("Enter matrix elements:\n");
    for (int i = 0; i < r; i++)
        for (int j = 0; j < c; j++)
            scanf("%d", &mat[i][j]);

    pthread_t *threads = (pthread_t*)malloc((r + c) * sizeof(pthread_t));
    int tCount = 0;

    for (int i = 0; i < r; i++) {
        int *data = (int*)malloc(2 * sizeof(int));
        data[0] = i; // row index
        data[1] = 0; // row type
        pthread_create(&threads[tCount++], NULL, calcSum, data);
    }

    for (int j = 0; j < c; j++) {
        int *data = (int*)malloc(2 * sizeof(int));
        data[0] = j; // column index
        data[1] = 1; // column type
        pthread_create(&threads[tCount++], NULL, calcSum, data);
    }

    for (int i = 0; i < tCount; i++)
        pthread_join(threads[i], NULL);

    printf("Row sums:\n");
    for (int i = 0; i < r; i++)
        printf("Row %d = %d\n", i, rowSum[i]);

    printf("Column sums:\n");
    for (int j = 0; j < c; j++)
        printf("Col %d = %d\n", j, colSum[j]);

    for (int i = 0; i < r; i++) free(mat[i]);
    free(mat); free(rowSum); free(colSum); free(threads);

    return 0;
}
