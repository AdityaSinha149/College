#include <stdio.h>
#include <stdlib.h>

void bfs(int **graph, int n, int start) {
    int *visited = (int *)malloc(n * sizeof(int));
    int *queue = (int *)malloc(n * sizeof(int));
    int front = 0, rear = 0;

    for (int i = 0; i < n; i++) visited[i] = 0;

    visited[start] = 1;
    queue[rear++] = start;

    while (front < rear) {
        int vertex = queue[front++];
        printf("%d ", vertex);

        for (int i = 0; i < n; i++) {
            if (graph[vertex][i] == 1 && !visited[i]) {
                visited[i] = 1;
                queue[rear++] = i;
            }
        }
    }

    free(visited);
    free(queue);
}

int main() {
    int n, start;

    printf("Enter the number of vertices in the graph: ");
    scanf("%d", &n);
    
    int **graph = (int **)malloc(n * sizeof(int *));
    
    for (int i = 0; i < n; i++) {
        graph[i] = (int *)malloc(n * sizeof(int));
    }

    printf("Enter the adjacency matrix for the graph (each row represents connections for a vertex):\n");
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%d", &graph[i][j]);
        }
    }
    
    printf("Enter the starting vertex for BFS traversal: ");
    scanf("%d", &start);
    
    printf("BFS traversal starting from vertex %d: ", start);
    bfs(graph, n, start);
    printf("\n");

    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
    
    return 0;
}
