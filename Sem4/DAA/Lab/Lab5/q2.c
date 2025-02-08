#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int val;
    struct Node* left;
    struct Node* right;
} Node;

Node* createNode(int val) {
    Node* newNode = (Node*)malloc(sizeof(Node));
    if (!newNode) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    newNode->val = val;
    newNode->left = newNode->right = NULL;
    return newNode;
}

Node* createBinaryTree() {
    int val;
    printf("Enter node value (enter -1 for no node): ");
    scanf("%d", &val);

    if (val == -1) return NULL;

    Node* newNode = createNode(val);

    printf("Enter left child of %d\n", val);
    newNode->left = createBinaryTree();

    printf("Enter right child of %d\n", val);
    newNode->right = createBinaryTree();

    return newNode;
}

int dfs(Node* root, int a) {
    if (root == NULL) return 0;
    if (root->val == a) return 1;
    return dfs(root->left, a) || dfs(root->right, a);
}

int cca(Node* root, int a, int b) {
    if (root == NULL) return -1;

    int left = dfs(root->left, a) + dfs(root->left, b);
    int right = dfs(root->right, a) + dfs(root->right, b);

    if (left + right == 1) return root->val;
    if (left == 2) return cca(root->left, a, b);
    if (right == 2) return cca(root->right, a, b);
    return -1;
}

int main() {
    Node* root = createBinaryTree();

    int a, b;
    printf("Enter two node values to find their lowest common ancestor: ");
    scanf("%d %d", &a, &b);

    int ancestor = cca(root, a, b);
    if (ancestor != -1)
        printf("Lowest Common Ancestor of %d and %d is %d\n", a, b, ancestor);
    else
        printf("No common ancestor found\n");

    free(root);
    return 0;
}
