#include <stdio.h>
#include <stdlib.h>

struct Edge {
    int u, v, weight;
};

int compareEdges(const void *a, const void *b) {
    return ((struct Edge *)a)->weight - ((struct Edge *)b)->weight;
}

int dfs(int current, int parent, int visited[], int graph[][100], int n) {
    visited[current] = 1;
    for (int i = 0; i < n; i++) {
        if (graph[current][i]) {
            if (!visited[i]) {
                if (dfs(i, current, visited, graph, n)) return 1;
            } else if (i != parent) return 1;
        }
    }
    return 0;
}

int createsCycle(int u, int v, int graph[][100], int n) {
    int visited[100] = {0};
    graph[u][v] = graph[v][u] = 1;
    int hasCycle = dfs(u, -1, visited, graph, n);
    graph[u][v] = graph[v][u] = 0;
    return hasCycle;
}

void kruskal(int n, int m, struct Edge *edges) {
    qsort(edges, m, sizeof(struct Edge), compareEdges);
    int graph[100][100] = {0};
    int mstWeight = 0, mstEdges = 0;

    for (int i = 0; i < m && mstEdges < n - 1; i++) {
        int u = edges[i].u, v = edges[i].v, weight = edges[i].weight;
        if (!createsCycle(u, v, graph, n)) {
            graph[u][v] = graph[v][u] = 1;
            printf("Edge (%d, %d) with weight %d\n", u, v, weight);
            mstWeight += weight;
            mstEdges++;
        }
    }
    printf("Total MST weight: %d\n", mstWeight);
}

int main() {
    int n, m;
    printf("Enter number of vertices and edges: ");
    scanf("%d %d", &n, &m);
    struct Edge edges[m];
    printf("Enter edges (u, v, weight):\n");
    for (int i = 0; i < m; i++) {
        scanf("%d %d %d", &edges[i].u, &edges[i].v, &edges[i].weight);
    }
    kruskal(n, m, edges);
    return 0;
}