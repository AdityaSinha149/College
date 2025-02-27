#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define MAX 20

int graph[MAX][MAX], n;

int isClique(int vertices[], int size) {
    for (int i = 0; i < size; i++) {
        for (int j = i + 1; j < size; j++) {
            if (graph[vertices[i]][vertices[j]] == 0)
                return 0;
        }
    }
    return 1;
}

void findMaxClique() {
    int maxSize = 0, maxClique[MAX];
    int totalSubsets = (1 << n);

    for (int subset = 1; subset < totalSubsets; subset++) {
        int vertices[MAX], size = 0;
        for (int i = 0; i < n; i++) {
            if (subset & (1 << i))
                vertices[size++] = i;
        }

        if (isClique(vertices, size) && size > maxSize) {
            maxSize = size;
            for (int i = 0; i < size; i++)
                maxClique[i] = vertices[i];
        }
    }

    printf("Maximal Clique of size %d: ", maxSize);
    for (int i = 0; i < maxSize; i++)
        printf("%d ", maxClique[i]);
    printf("\n");
}

int main() {
    printf("Enter number of vertices: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    findMaxClique();
    return 0;
}
