#include <stdio.h>
#include <stdlib.h>

#define MAX 100

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct {
    Node* head[MAX];
    int matrix[MAX][MAX];
    int vertices;
} Graph;

Graph* createGraph(int vertices) {
    Graph* graph = (Graph*)malloc(sizeof(Graph));
    graph->vertices = vertices;
    for (int i = 0; i < vertices; i++) {
        graph->head[i] = NULL;
        for (int j = 0; j < vertices; j++) graph->matrix[i][j] = 0;
    }
    return graph;
}

void addEdge(Graph* graph, int src, int dest) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->vertex = dest;
    newNode->next = graph->head[src];
    graph->head[src] = newNode;
    graph->matrix[src][dest] = 1;
    graph->matrix[dest][src] = 1;
}

void displayAdjList(Graph* graph) {
    for (int i = 0; i < graph->vertices; i++) {
        printf("%d: ", i);
        Node* temp = graph->head[i];
        while (temp) {
            printf("%d ", temp->vertex);
            temp = temp->next;
        }
        printf("\n");
    }
}

void displayAdjMatrix(Graph* graph) {
    for (int i = 0; i < graph->vertices; i++) {
        for (int j = 0; j < graph->vertices; j++)
            printf("%d ", graph->matrix[i][j]);
        printf("\n");
    }
}

void freeGraph(Graph* graph) {
    for (int i = 0; i < graph->vertices; i++) {
        Node* temp = graph->head[i];
        while (temp) {
            Node* toFree = temp;
            temp = temp->next;
            free(toFree);
        }
    }
    free(graph);
}

int main() {
    int vertices, edges, src, dest;
    printf("Enter number of vertices: ");
    scanf("%d", &vertices);
    Graph* graph = createGraph(vertices);
    printf("Enter number of edges: ");
    scanf("%d", &edges);
    for (int i = 0; i < edges; i++) {
        printf("Enter edge (src dest): ");
        scanf("%d %d", &src, &dest);
        addEdge(graph, src, dest);
    }
    printf("\nAdjacency List:\n");
    displayAdjList(graph);
    printf("\nAdjacency Matrix:\n");
    displayAdjMatrix(graph);
    freeGraph(graph);
    return 0;
}
