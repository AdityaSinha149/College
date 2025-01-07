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

// Insert element into the back of the queue
void insertBack(queue* Q, int val) {
    if (isFull(Q)) {
        printf("Queue overflow\n");
        return;
    }
    Q->arr[++Q->r] = val;
}

// Insert element into the front of the queue
void insertFront(queue* Q, int val) {
    if (Q->f == -1) {
        printf("No space at front\n");
        return;
    }
    Q->arr[Q->f--] = val;
}

// Remove (poll) element from the front of the queue
int pollFront(queue* Q) {
    if (isEmpty(Q)) {
        printf("Queue underflow\n");
        return -1;
    }
    return Q->arr[++Q->f];  // Increment front and return the element
}

// Remove (poll) element from the back of the queue
int pollBack(queue* Q) {
    if (isEmpty(Q)) {
        printf("Queue underflow\n");
        return -1;
    }
    return Q->arr[Q->r--];
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

    insertBack(&q, 10);
    insertBack(&q, 20);
    insertBack(&q, 30);

    printf("Queue after insertions: ");
    display(&q);

    printf("Polled front: %d\n", pollFront(&q));
    printf("Queue after polling front: ");
    display(&q);

    insertFront(&q, 5);  // Attempting to insert at front (won't work because there's no space)
    printf("Queue after insert front: ");
    display(&q);

    printf("Polled back: %d\n", pollBack(&q));
    printf("Queue after polling back: ");
    display(&q);

    return 0;
}
