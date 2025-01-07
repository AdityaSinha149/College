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

void insertInBT(node* root, char arr[], int ele) {
    node *temp = createNode(ele);
    node *curr = root, *parent = NULL;
    int i = 0;

    while (curr && arr[i] != '\0') {
        parent = curr;
        if (arr[i] == 'l' || arr[i] == 'L')
            curr = curr->left;
        else if (arr[i] == 'r' || arr[i] == 'R')
            curr = curr->right;
        else {
            printf("Invalid direction character in path\n");
            free(temp);
            return;
        }
        i++;
    }

    // Check if we reached a valid position to insert
    if (curr != NULL || arr[i] != '\0') {
        free(temp);
        printf("Invalid operation: Path does not reach an empty node\n");
        return;
    }

    // Insert the node as left or right child based on last character in path
    if (arr[i-1] == 'l' || arr[i-1] == 'L')
        parent->left = temp;
    else
        parent->right = temp;
}

// Helper function to display the tree in inorder traversal
void inorderTraversal(node* root) {
    if (root == NULL) return;
    inorderTraversal(root->left);
    printf("%d -> ", root->val);
    inorderTraversal(root->right);
}

int main() {
    node* root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);

    printf("Tree before insertion (Inorder): ");
    inorderTraversal(root);
    printf("\n");

    insertInBT(root, "LLR", 6);

    printf("Tree after insertion of 6 at path 'LLR' (Inorder): ");
    inorderTraversal(root);
    printf("\n");

    return 0;
}
