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

node* findMin(node* root) {
    while (root->left != NULL)
        root = root->left;
    return root;
}

node* deleteNode(node* root, int key) {
    if (root == NULL)
        return root;

    if (key < root->val)
        root->left = deleteNode(root->left, key);
    else if (key > root->val)
        root->right = deleteNode(root->right, key);
    else {
        // Node found
        // Case 1: Node with only one child or no child
        if (root->left == NULL) {
            node* temp = root->right;
            free(root);
            return temp;
        } else if (root->right == NULL) {
            node* temp = root->left;
            free(root);
            return temp;
        }

        // Case 2: Node with two children
        node* temp = findMin(root->right); // Get the inorder successor
        root->val = temp->val;
        root->right = deleteNode(root->right, temp->val);
    }
    return root;
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

    printf("Inorder traversal of the tree:\n");
    display(root);
    printf("NULL\n");

    int key;
    printf("Enter value to delete from tree: ");
    scanf("%d", &key);
    root = deleteNode(root, key);

    printf("Inorder traversal after deletion:\n");
    display(root);
    printf("NULL\n");

    return 0;
}
