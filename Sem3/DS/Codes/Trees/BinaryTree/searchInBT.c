#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    int val;
    struct node* left;
    struct node* right;
} node;

node* createNode(int value) {
    node* newnode = (node*)malloc(sizeof(node));
    newnode->val = value;
    newnode->left = newnode->right = NULL;
    return newnode;
}

int searchInBT(node* root, int key) {
    if (root == NULL) return 0;
    if (root->val == key) return 1;
    // Recursively search in left or right subtree
    return searchInBT(root->left, key) || searchInBT(root->right, key);
}

int main() {
    node* root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);

    int key = 4;
    if (searchInBT(root, key))
        printf("Element %d found in the binary tree.\n", key);
    else
        printf("Element %d not found in the binary tree.\n", key);

    key = 25;
    if (searchInBT(root, key))
        printf("Element %d found in the binary tree.\n", key);
    else
        printf("Element %d not found in the binary tree.\n", key);

    return 0;
}
