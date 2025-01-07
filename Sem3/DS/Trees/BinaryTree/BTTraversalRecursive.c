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

void inorder(node* root) {
    if (root) {
        inorder(root->left);
        printf("%d -> ", root->val);
        inorder(root->right);
    }
}

void postorder(node* root) {
    if (root) {
        postorder(root->left);
        postorder(root->right);
        printf("%d -> ", root->val);
    }
}

void preorder(node* root) {
    if (root) {
        printf("%d -> ", root->val);
        preorder(root->left);
        preorder(root->right);
    }
}

int main() {
    node* root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);

    printf("Inorder traversal: ");
    inorder(root);
    printf("NULL\n");

    printf("Preorder traversal: ");
    preorder(root);
    printf("NULL\n");

    printf("Postorder traversal: ");
    postorder(root);
    printf("NULL\n");

    return 0;
}
