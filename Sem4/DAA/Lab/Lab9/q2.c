#include <stdio.h>
#include <stdlib.h>

int opcount = 0;

typedef struct node {
    int data;
    struct node *next;
} node;

node* createnode(int key) {
    node *newnode = (node*)malloc(sizeof(node));
    newnode->data = key;
    newnode->next = NULL;
    return newnode;
}

void insert(node* hashmap[], int m, int key) {
    int index = key % m;
    node* newnode = createnode(key);

    if (hashmap[index] == NULL)
        hashmap[index] = newnode;
    else {
        node* temp = hashmap[index];
        while (temp->next != NULL)
            temp = temp->next;
        temp->next = newnode;
    }
}

int search(node* hashmap[], int m, int key) {
    int index = key % m;
    node* temp = hashmap[index];
    while (temp != NULL) {
        opcount++;
        if (temp->data == key)
            return 1;
        temp = temp->next;
    }
    return 0;
}

int main() {
    int m;
    printf("Enter the size of the hash table: ");
    scanf("%d", &m);

    node** hashmap = (node**)malloc(m * sizeof(node*));
    for (int i = 0; i < m; i++)
        hashmap[i] = NULL;

    while (1) {
        int key;
        printf("Enter the integer to insert (-1 to stop): ");
        scanf("%d", &key);
        if (key == -1)
            break;
        insert(hashmap, m, key);
    }

    int key;
    printf("Enter the integer to search: ");
    scanf("%d", &key);
    if (search(hashmap, m, key))
        printf("Element found\n");
    else
        printf("Element not found\n");

    for (int i = 0; i < m; i++) {
        node* temp = hashmap[i];
        while (temp != NULL) {
            node* toDelete = temp;
            temp = temp->next;
            free(toDelete);
        }
    }
    free(hashmap);
    
    printf("opcount:%d",opcount);

    return 0;
}
