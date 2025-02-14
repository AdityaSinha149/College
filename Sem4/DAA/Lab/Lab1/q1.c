#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int val;
    struct Node* left;
    struct Node* right;
} Node;

Node* createNode(int val) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    newNode->val = val;
    newNode->left = newNode->right = NULL;
    return newNode;
}

Node* insert(Node* root, int key) {
    if (!root) return createNode(key);
    if (key < root->val) root->left = insert(root->left, key);
    else if (key > root->val) root->right = insert(root->right, key);
    return root;
}

int search(Node* root, int key) {
    if (!root) return 0;
    if (root->val == key) return 1;
    return search(key < root->val ? root->left : root->right, key);
}

void inorder(Node* root) {
    if (!root) return;
    inorder(root->left);
    printf("%d ", root->val);
    inorder(root->right);
}

void preorder(Node* root) {
    if (!root) return;
    printf("%d ", root->val);
    preorder(root->left);
    preorder(root->right);
}

void postorder(Node* root) {
    if (!root) return;
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->val);
}

void freeTree(Node* root) {
    if (!root) return;
    freeTree(root->left);
    freeTree(root->right);
    free(root);
}

int main() {
    Node* root = NULL;
    int key, choice;
    while (1) {
        printf("\n1. Search/Insert\n2. Inorder\n3. Preorder\n4. Postorder\n5. Exit\nEnter choice: ");
        scanf("%d", &choice);
        if (choice == 1) {
            printf("Enter key: ");
            scanf("%d", &key);
            if (search(root, key)) printf("key found\n");
            else root = insert(root, key);
        } else if (choice == 2) inorder(root);
        else if (choice == 3) preorder(root);
        else if (choice == 4) postorder(root);
        else break;
    }
    freeTree(root);
    return 0;
}
