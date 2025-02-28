#include <stdio.h>
#include <stdlib.h>

#define MAX 100

int graph[MAX][MAX];  
int visited[MAX];  
int path[MAX];     
int vertices, edges;

void addEdge(int u, int v) {
    graph[u][v] = 1;
    graph[v][u] = 1;
}

void printPath(int pathLength) {
    printf("Path to exit: ");
    for (int i = 0; i < pathLength; i++) 
        printf("%d -> ", path[i]);
    printf("EXIT\n");
}

int dfs(int current, int end, int pathLength) {
    visited[current] = 1;
    path[pathLength++] = current;

    if (current == end) {
        printPath(pathLength);
        return 1;
    }

    for (int i = 0; i < vertices; i++) {
        if (graph[current][i] == 1 && !visited[i]) {
            if (dfs(i, end, pathLength)) 
                return 1;
        }
    }
    
    return 0;
}

int main() {
    int start, end;
    
    printf("Enter number of maze junctions: ");
    scanf("%d", &vertices);
    
    printf("Enter number of paths: ");
    scanf("%d", &edges);

    printf("Enter %d paths (start end):\n", edges);
    for (int i = 0; i < edges; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        addEdge(u, v);
    }

    printf("Enter start and exit junctions: ");
    scanf("%d %d", &start, &end);

    if (!dfs(start, end, 0)) 
        printf("No path found!\n");

    return 0;
}
