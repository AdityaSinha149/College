#include <stdio.h>
#include <stdlib.h>

int *visited;
int *push_order, *pop_order;
int push_idx = 0, pop_idx = 0;

void dfs(int **graph, int vertex, int n) {
    visited[vertex] = 1;
    push_order[push_idx++] = vertex;
    
    for (int i = 0; i < n; i++) {
        if (graph[vertex][i] == 1 && !visited[i]) {
            dfs(graph, i, n);
        }
    }
    
    pop_order[pop_idx++] = vertex;
}

int main() {
    int n, start;
    
    printf("Enter the number of vertices in the graph: ");
    scanf("%d", &n);
    
    int **graph = (int **)malloc(n * sizeof(int *));
    visited = (int *)malloc(n * sizeof(int));
    push_order = (int *)malloc(n * sizeof(int));
    pop_order = (int *)malloc(n * sizeof(int));

    // Initialize visited array to 0 (not visited)
    for (int i = 0; i < n; i++) {
        graph[i] = (int *)malloc(n * sizeof(int));
        visited[i] = 0;
    }

    printf("Enter the adjacency matrix (a %d x %d matrix):\n", n, n);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);
    
    printf("Enter the starting vertex (0-based index): ");
    scanf("%d", &start);
    
    printf("\nRunning Depth-First Search (DFS) starting from vertex %d...\n\n", start);
    
    dfs(graph, start, n);
    
    printf("Push order (the order vertices are visited):\n");
    for (int i = 0; i < push_idx; i++) printf("%d ", push_order[i]);
    printf("\n");
    
    printf("Pop order (the order vertices are finished visiting):\n");
    for (int i = 0; i < pop_idx; i++) printf("%d ", pop_order[i]);
    printf("\n");
    
    // Free dynamically allocated memory
    for (int i = 0; i < n; i++) {
        free(graph[i]);
    }
    free(graph);
    free(visited);
    free(push_order);
    free(pop_order);
    
    return 0;
}
