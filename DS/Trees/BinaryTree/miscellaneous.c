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

//Calculate the height of the binary tree
int heightOfBT(node* root) {
    if (root == NULL) return 0;
    int leftHeight = heightOfBT(root->left);
    int rightHeight = heightOfBT(root->right);
    return 1 + (leftHeight > rightHeight ? leftHeight : rightHeight);
}

//Calculate the number of nodes in the binary tree
int countNodes(node* root) {
    if (root == NULL) return 0;
    return 1 + countNodes(root->left) + countNodes(root->right);
}

//Calculate the number of leaf nodes in the binary tree
int countLeafNodes(node* root) {
    if (root == NULL) return 0;
    if (root->left == NULL && root->right == NULL) return 1;
    return countLeafNodes(root->left) + countLeafNodes(root->right);
}

int main() {
    node* root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);

    printf("Height of the binary tree: %d\n", heightOfBT(root));
    printf("Number of nodes in the binary tree: %d\n", countNodes(root));
    printf("Number of leaf nodes in the binary tree: %d\n", countLeafNodes(root));
}
