#include <stdio.h>
#include <stdlib.h>

#define MAX 100  

int adj[MAX][MAX], inDegree[MAX], n;

void findSchedule() {
    int queue[MAX], front = 0, rear = 0, count = 0;

    printf("Execution Order: ");
    for (int i = 0; i < n; i++)
        if (inDegree[i] == 0)
            queue[rear++] = i;

    while (front < rear) {
        int module = queue[front++];
        printf("%d ", module);
        count++;

        for (int i = 0; i < n; i++)
            if (adj[module][i] && --inDegree[i] == 0)
                queue[rear++] = i;
    }

    if (count != n)
        printf("\nCycle detected! Some tasks have circular dependencies.\n");
    else
        printf("\nNo cycle detected. All tasks can be completed.\n");
}

int main() {
    int edges;
    printf("Enter the number of modules: ");
    scanf("%d", &n);
    
    printf("Enter the number of dependencies: ");
    scanf("%d", &edges);

    printf("Enter dependencies (module A depends on module B as: A B):\n");
    for (int i = 0; i < edges; i++) {
        int a, b;
        scanf("%d %d", &a, &b);
        adj[b][a] = 1;
        inDegree[a]++;
    }

    findSchedule();
    return 0;
}
