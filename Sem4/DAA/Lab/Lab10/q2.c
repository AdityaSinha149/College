#include <stdio.h>
#include <stdlib.h>
#include <limits.h>

#define INF INT_MAX

void floydWarshall(int** graph, int n) {
    for (int k = 0; k < n; k++) {
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                if (graph[i][k] != INF && graph[k][j] != INF && graph[i][j] > graph[i][k] + graph[k][j]) {
                    graph[i][j] = graph[i][k] + graph[k][j];
                }
            }
        }
    }

    printf("The shortest graphance matrix is:\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (graph[i][j] == INF) {
                printf("INF ");
            } else {
                printf("%d ", graph[i][j]);
            }
        }
        printf("\n");
    }

}

int main() {
    int n;
    printf("Enter the number of vertices: ");
    scanf("%d", &n);

    int** graph = (int**)malloc(n * sizeof(int*));
    for (int i = 0; i < n; i++) {
        graph[i] = (int*)malloc(n * sizeof(int));
    }

    printf("Enter the adjacency matrix (%d x %d) (use %d for INF):\n", n, n, INF);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &graph[i][j]);
        }
    }

    floydWarshall(graph, n);

    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);

    return 0;
}