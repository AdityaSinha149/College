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

Node* cca(Node* root, int a, int b) {
    if (root == NULL) return NULL;

    if (root->val == a || root->val == b) return root;

    Node* leftLCA = cca(root->left, a, b);
    Node* rightLCA = cca(root->right, a, b);
    if (leftLCA != NULL && rightLCA != NULL) return root;

    return (leftLCA != NULL) ? leftLCA : rightLCA;
}

void freeTree(Node* root) {
    if (root == NULL) return;
    freeTree(root->left);
    freeTree(root->right);
    free(root);
}

int main() {
    Node* root = createBinaryTree();

    int a, b;
    while (1) {
        printf("Enter two node values to find their lowest common ancestor: ");
        scanf("%d %d", &a, &b);

        Node* ancestor = cca(root, a, b);
        if (ancestor != NULL)
            printf("Lowest Common Ancestor of %d and %d is %d\n", a, b, ancestor->val);
        else
            printf("No common ancestor found\n");
    }
    freeTree(root);

    return 0;
}
