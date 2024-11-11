#include <stdio.h>
#include <stdlib.h>

// Define Node structure
typedef struct Node {
    int val;
    struct Node* next;
} Node;

// Initialize list with a single node
Node* initializeList(int val) {
    Node* head = (Node*)malloc(sizeof(Node));
    head->val = val;
    head->next = NULL;
    return head;
}

// Insert node at the front
void insertFront(Node **h, int val) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->val = val;
    newNode->next = *h;
    *h = newNode;
}

// Delete node at the front
void deleteFront(Node** h) {
    if (*h == NULL) return;
    Node* temp = *h;
    *h = (*h)->next;
    free(temp);
}

// Insert node at the end
void insertEnd(Node** h, int val) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->val = val;
    newNode->next = NULL;

    if (*h == NULL) { // If list is empty, set new node as head
        *h = newNode;
        return;
    }

    Node* p = *h;
    while (p->next != NULL) p = p->next;
    p->next = newNode;
}

// Delete node at the end
void deleteEnd(Node** h) {
    if (*h == NULL) return;
    
    Node* p = *h;
    if (p->next == NULL) { // If there's only one node
        free(p);
        *h = NULL;
        return;
    }

    while (p->next->next != NULL) p = p->next;
    
    Node* temp = p->next;
    p->next = NULL;
    free(temp);
}

// Insert node in the middle
void insertMiddle(Node** h, int val, int idx) {
    if (idx == 0) {
        insertFront(h, val);
        return;
    }

    Node* p = *h;
    while (idx > 1 && p->next != NULL) {
        p = p->next;
        idx--;
    }

    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->val = val;
    newNode->next = p->next;
    p->next = newNode;
}

// Delete node in the middle
void deleteMiddle(Node** h, int idx) {
    if (*h == NULL || idx == 0) {
        deleteFront(h);
        return;
    }

    Node* p = *h;
    Node* prev = NULL;
    while (idx > 0 && p->next != NULL) {
        prev = p;
        p = p->next;
        idx--;
    }

    if (prev != NULL && p != NULL) {
        prev->next = p->next;
        free(p);
    }
}

// Display the list
void display(Node* h) {
    Node* p = h;
    while (p != NULL) {
        printf("%d -> ", p->val);
        p = p->next;
    }
    printf("NULL\n");
}

// Main function
int main() {
    Node* head = initializeList(10); // Initialize list

    insertFront(&head, 5);           // Insert at front
    insertEnd(&head, 20);            // Insert at end
    insertMiddle(&head, 15, 1);      // Insert in middle

    printf("List after insertions: ");
    display(head);                   // Display list

    deleteFront(&head);              // Delete from front
    deleteEnd(&head);                // Delete from end
    deleteMiddle(&head, 1);          // Delete from middle

    printf("List after deletions: ");
    display(head);                   // Display list

    return 0;
}
