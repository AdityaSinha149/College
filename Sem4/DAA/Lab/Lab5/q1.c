#include <stdio.h>
#include <stdlib.h>

int *topo, *visited, idx = 0;

void dfs(int** graph, int vertex, int n) {
    visited[vertex] = 1;
    for (int i = 0; i < n; i++) {
        if (graph[vertex][i] && !visited[i]) {
            dfs(graph, i, n);
        }
    }
    topo[idx++] = vertex;
}

int main() {
    int n, m;

    printf("Enter the number of vertices: ");
    scanf("%d", &n);

    int **graph = (int**)malloc(n * sizeof(int*));
    visited = (int*)malloc(n * sizeof(int));
    topo = (int*)malloc(n * sizeof(int));

    for (int i = 0; i < n; i++) {
        graph[i] = (int*)malloc(n * sizeof(int));
    }

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            graph[i][j] = 0;
        }
    }

    printf("Enter the number of edges: ");
    scanf("%d", &m);

    printf("Enter the edges (vertex pair from and to):\n");
    for (int i = 0; i < m; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        graph[u][v] = 1;
    }

    for (int i = 0; i < n; i++) {
        if (!visited[i]) {
            dfs(graph, i, n);
        }
    }

    printf("Topological Sort: ");
    for (int i = n - 1; i >= 0; i--) {
        printf("%d ", topo[i]);
    }
    printf("\n");

    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
    free(visited);
    free(topo);

    return 0;
}
