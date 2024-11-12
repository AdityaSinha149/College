#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    int val;
    struct node* prev;
    struct node* next;
} node;

node* createNode(int x) {
    node* newnode = (node*)malloc(sizeof(node));
    newnode->val = x;
    newnode->prev = newnode->next = NULL;
    return newnode;
}

node* insertAtEnd(node* head, int x) {
    node* newnode = createNode(x);
    if (head == NULL)
        return newnode;

    node* temp = head;
    while (temp->next != NULL)
        temp = temp->next;

    temp->next = newnode;
    newnode->prev = temp;
    return head;
}

node* insertAtBeg(node* head, int x) {
    node* newnode = createNode(x);
    if (head == NULL)
        return newnode;

    newnode->next = head;
    head->prev = newnode;
    return newnode;
}

node* insertInMiddle(node* head, int x, int n) {
    if (head == NULL)
        return insertAtBeg(head, x);

    node* temp = head;
    for (int i = 0; i < n - 1; i++) {
        if (temp->next == NULL) {
            printf("Position not found\n");
            return head;
        }
        temp = temp->next;
    }

    if (temp->next == NULL)
        return insertAtEnd(head, x);

    node* newnode = createNode(x);
    newnode->next = temp->next;
    newnode->prev = temp;
    temp->next->prev = newnode;
    temp->next = newnode;
    return head;
}

node* deleteAtEnd(node* head) {
    if (head == NULL) {
        printf("List is empty\n");
        return head;
    }

    if (head->next == NULL) {
        free(head);
        return NULL;
    }

    node* temp = head;
    while (temp->next != NULL)
        temp = temp->next;

    temp->prev->next = NULL;
    free(temp);
    return head;
}

node* deleteAtBeg(node* head) {
    if (head == NULL) {
        printf("List is empty\n");
        return head;
    }

    node* temp = head;
    head = head->next;
    if (head != NULL)
        head->prev = NULL;

    free(temp);
    return head;
}

node* deleteAtMiddle(node* head, int n) {
    if (head == NULL) {
        printf("List is empty\n");
        return head;
    }

    if (n == 1)
        return deleteAtBeg(head);

    node* temp = head;
    for (int i = 0; i < n - 1; i++) {
        if (temp->next == NULL) {
            printf("Position not found\n");
            return head;
        }
        temp = temp->next;
    }

    if (temp->next == NULL)
        return deleteAtEnd(head);

    temp->prev->next = temp->next;
    temp->next->prev = temp->prev;
    free(temp);
    return head;
}

void display(node* head) {
    if (head == NULL) {
        printf("List is empty\n");
        return;
    }

    node* temp = head;
    while (temp != NULL) {
        printf("%d -> ", temp->val);
        temp = temp->next;
    }
    printf("NULL\n");
}

int main() {
    node* head = NULL;
    int choice, value, position;

    do {
        printf("\nMenu:\n");
        printf("1. Insert at Beginning\n");
        printf("2. Insert at End\n");
        printf("3. Insert at Position\n");
        printf("4. Delete from Beginning\n");
        printf("5. Delete from End\n");
        printf("6. Delete from Position\n");
        printf("7. Display List\n");
        printf("8. Exit\n");
        printf("Enter your choice: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1:
                printf("Enter value to insert at beginning: ");
                scanf("%d", &value);
                head = insertAtBeg(head, value);
                break;

            case 2:
                printf("Enter value to insert at end: ");
                scanf("%d", &value);
                head = insertAtEnd(head, value);
                break;

            case 3:
                printf("Enter value to insert: ");
                scanf("%d", &value);
                printf("Enter position to insert at: ");
                scanf("%d", &position);
                head = insertInMiddle(head, value, position);
                break;

            case 4:
                head = deleteAtBeg(head);
                break;

            case 5:
                head = deleteAtEnd(head);
                break;

            case 6:
                printf("Enter position to delete from: ");
                scanf("%d", &position);
                head = deleteAtMiddle(head, position);
                break;

            case 7:
                display(head);
                break;

            case 8:
                printf("Exiting program...\n");
                break;

            default:
                printf("Invalid choice. Please try again.\n");
        }
    } while (choice != 8);

    return 0;
}
