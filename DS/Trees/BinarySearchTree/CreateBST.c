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

int main() {
    node* root = NULL;
    createBST(&root);
    printf("Inorder traversal of the given tree: \n");
    display(root);
    printf("NULL\n");
    return 0;
}
