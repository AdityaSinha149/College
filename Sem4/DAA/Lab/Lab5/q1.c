#include <stdio.h>
#include <stdlib.h>

void dfs(int** graph, int vertex, int n, int* visited, int* topo, int* idx, int* cycleFlag) {
    if (visited[vertex] == 1) {
        *cycleFlag = 1;
        return;
    }
    if (visited[vertex] == 2) return;

    visited[vertex] = 1;

    for (int i = 0; i < n; i++) {
        if (graph[vertex][i] && visited[i] != 2) {
            dfs(graph, i, n, visited, topo, idx, cycleFlag);
        }
    }

    visited[vertex] = 2;
    topo[(*idx)++] = vertex;
}

void sourceRemoval(int **graph, int *inDegree, int n) {
    int *queue = (int*)malloc(n * sizeof(int));
    int front = 0, rear = 0;
    int idx = 0;

    for (int i = 0; i < n; i++) {
        if (inDegree[i] == 0) {
            queue[rear++] = i;
        }
    }

    while (front < rear) {
        int node = queue[front++];

        printf("%d ", node);
        idx++;

        for (int i = 0; i < n; i++) {
            if (graph[node][i] == 1) {
                inDegree[i]--;
                if (inDegree[i] == 0) {
                    queue[rear++] = i;
                }
            }
        }
    }

    if (idx != n) {
        printf("\nGraph has a cycle.\n");
    } else {
        printf("\n");
    }

    free(queue);
}

int main() {
    int n, m;
    int cycleFlag = 0;

    printf("Enter the number of vertices: ");
    scanf("%d", &n);

    int **graph = (int**)malloc(n * sizeof(int*));
    int *visited = (int*)malloc(n * sizeof(int));
    int *topo = (int*)malloc(n * sizeof(int));
    int *inDegree = (int*)malloc(n * sizeof(int));
    int idx = 0;

    for (int i = 0; i < n; i++) {
        graph[i] = (int*)malloc(n * sizeof(int));
        inDegree[i] = 0;
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
        inDegree[v]++;
    }

    for (int i = 0; i < n; i++) {
        visited[i] = 0;
    }

    for (int i = 0; i < n; i++) {
        if (visited[i] == 0) {
            dfs(graph, i, n, visited, topo, &idx, &cycleFlag);
            if (cycleFlag) break;
        }
    }

    printf("Topological Sort (DFS): ");
    if (cycleFlag) {
        printf("Graph has a cycle.\n");
    } else {
        for (int i = n - 1; i >= 0; i--) {
            printf("%d ", topo[i]);
        }
        printf("\n");
    }

    printf("Topological Sort (Source Removal Method): ");
    sourceRemoval(graph, inDegree, n);

    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
    free(inDegree);
    free(visited);
    free(topo);

    return 0;
}

