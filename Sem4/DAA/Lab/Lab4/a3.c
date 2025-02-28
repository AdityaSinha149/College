#include <stdio.h>
#include <limits.h>

#define MAX 10 

int n;
int graph[MAX][MAX];
int minCost = INT_MAX;
int bestPath[MAX];

void swap(int* a, int* b) {
    *a ^= *b;
    *b ^= *a;
    *a ^= *b;
}

void calculateCost(int path[]) {
    int cost = 0;
    for (int i = 0; i < n - 1; i++) 
        cost += graph[path[i]][path[i + 1]];
    
    cost += graph[path[n - 1]][path[0]];

    if (cost < minCost) {
        minCost = cost;
        for (int i = 0; i < n; i++)
            bestPath[i] = path[i];
    }
}

void generatePermutations(int path[], int l, int r) {
    if (l == r) {
        calculateCost(path);
        return;
    }

    for (int i = l; i <= r; i++) {
        swap(&path[l], &path[i]);
        generatePermutations(path, l + 1, r);
        swap(&path[l], &path[i]);
    }
}

int main() {
    printf("Enter number of cities: ");
    scanf("%d", &n);

    printf("Enter adjacency matrix:\n");
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            scanf("%d", &graph[i][j]);

    int path[MAX];
    for (int i = 0; i < n; i++)
        path[i] = i;

    generatePermutations(path, 1, n - 1);

    printf("Minimum cost: %d\n", minCost);
    printf("Optimal path: ");
    for (int i = 0; i < n; i++)
        printf("%d -> ", bestPath[i]);
    printf("%d\n", bestPath[0]);

    return 0;
}
