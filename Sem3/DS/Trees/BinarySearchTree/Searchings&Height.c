#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    int val;
    struct node* left;
    struct node* right;
} node;

node* createNode(int x) {
    node* newnode = (node*)malloc(sizeof(node));
    newnode->val = x;
    newnode->left = newnode->right = NULL;
    return newnode;
}

void insertBST(node** root, int x) {
    if (*root == NULL) {
        *root = createNode(x);
        return;
    }
    if (x < (*root)->val)
        insertBST(&(*root)->left, x);
    else
        insertBST(&(*root)->right, x);
}

void createBST(node** root) {
    int x;
    printf("Enter values to insert into tree (-1 to exit):\n");
    while (1) {
        scanf("%d", &x);
        if (x == -1) break;
        insertBST(root, x);
    }
}

void display(node* root) {
    if (root != NULL) {
        display(root->left);
        printf("%d -> ", root->val);
        display(root->right);
    }
}

// Recursive Search Function
node* searchRecursive(node* root, int key) {
    if (root == NULL || root->val == key)
        return root;
    if (key < root->val)
        return searchRecursive(root->left, key);
    return searchRecursive(root->right, key);
}

// Iterative Search Function
node* searchIterative(node* root, int key) {
    while (root != NULL && root->val != key) {
        if (key < root->val)
            root = root->left;
        else
            root = root->right;
    }
    return root;
}

// Find minimum value in the tree
node* findMin(node* root) {
    while (root->left != NULL)
        root = root->left;
    return root;
}

// Find maximum value in the tree
node* findMax(node* root) {
    while (root->right != NULL)
        root = root->right;
    return root;
}

// Calculate height of the tree
int max(int a, int b) {
    return (a > b) ? a : b;
}

int findHeight(node* root) {
    if (root == NULL)
        return 0;
    return 1 + max(findHeight(root->left), findHeight(root->right));
}

int main() {
    node* root = NULL;
    createBST(&root);

    printf("Inorder traversal of the given tree: \n");
    display(root);
    printf("NULL\n");

    int height = findHeight(root);
    printf("Height of the tree: %d\n", height);

    node* minNode = findMin(root);
    printf("Minimum value in the tree: %d\n", minNode->val);

    node* maxNode = findMax(root);
    printf("Maximum value in the tree: %d\n", maxNode->val);

    return 0;
}
