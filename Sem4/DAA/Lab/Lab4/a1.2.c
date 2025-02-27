#include <stdio.h>
#include <stdlib.h>

#define MAX 100  

int adj[MAX][MAX], color[MAX], n;

int isBipartiteBFS(int start) {
    int queue[MAX], front = 0, rear = 0;
    queue[rear++] = start;
    color[start] = 0;

    while (front < rear) {
        int node = queue[front++];

        for (int i = 0; i < n; i++) {
            if (adj[node][i]) {
                if (color[i] == -1) {
                    color[i] = 1 - color[node];
                    queue[rear++] = i;
                } else if (color[i] == color[node]) 
                    return 0;
            }
        }
    }
    return 1;
}

int main() {
    int edges;
    printf("Enter number of vertices: ");
    scanf("%d", &n);
    
    printf("Enter number of edges: ");
    scanf("%d", &edges);

    printf("Enter edges (format: u v):\n");
    for (int i = 0; i < edges; i++) {
        int u, v;
        scanf("%d %d", &u, &v);
        adj[u][v] = adj[v][u] = 1;
    }

    for (int i = 0; i < n; i++) 
        color[i] = -1;

    int bipartite = 1;
    for (int i = 0; i < n; i++) {
        if (color[i] == -1) {
            if (!isBipartiteBFS(i)) {
                bipartite = 0;
                break;
            }
        }
    }

    if (bipartite)
        printf("The graph is Bipartite\n");
    else
        printf("The graph is NOT Bipartite\n");

    return 0;
}
