#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

typedef struct {
    int* arr;
    int f;
    int r;
    int size;
} queue;

// Initialize the queue
queue initializeQueue(int size) {
    queue Q;
    Q.arr = (int*)malloc(size * sizeof(int));
    Q.f = -1;
    Q.r = -1;
    Q.size = size;
    return Q;
}

// Check if the queue is empty
bool isEmpty(queue* Q) {
    return Q->f == Q->r;
}

// Check if the queue is full
bool isFull(queue* Q) {
    return Q->r == Q->size - 1;
}

// Insert element into the queue
void insert(queue* Q, int val) {
    if (isFull(Q)) {
        printf("Queue overflow\n");
        return;
    }
    Q->arr[++Q->r] = val;
}

// Remove (poll) element from the queue
int poll(queue* Q) {
    if (isEmpty(Q)) {
        printf("Queue underflow\n");
        return -1;
    }
    return Q->arr[++Q->f];  // Increment front and return the element
}

// Display the queue
void display(queue* Q) {
    if (isEmpty(Q)) {
        printf("Queue is empty\n");
        return;
    }

    for (int i = Q->f + 1; i <= Q->r; i++) {
        printf("%d ", Q->arr[i]);
    }
    printf("\n");
}

// Main function to demonstrate usage
int main() {
    queue q = initializeQueue(5);

    insert(&q, 10);
    insert(&q, 20);
    insert(&q, 30);

    printf("Queue after insertions: ");
    display(&q);

    printf("Polled: %d\n", poll(&q));
    printf("Queue after polling: ");
    display(&q);

    return 0;
}
