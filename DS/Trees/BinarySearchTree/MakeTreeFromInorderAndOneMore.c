#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    int val;
    struct node* left;
    struct node* right;
} node;

// Function to create a new node
node* createNode(int x) {
    node* newNode = (node*)malloc(sizeof(node));
    newNode->val = x;
    newNode->left = newNode->right = NULL;
    return newNode;
}

// Helper function to find the index of a value in an array
int search(int arr[], int start, int end, int value) {
    for (int i = start; i <= end; i++) {
        if (arr[i] == value) {
            return i;
        }
    }
    return -1;
}

// Recursive function to construct binary tree from inorder and preorder traversals
node* buildTreeFromInorderPreorder(int inorder[], int preorder[], int inorderStart, int inorderEnd, int* preorderIndex) {
    if (inorderStart > inorderEnd) return NULL;

    int rootValue = preorder[*preorderIndex];
    (*preorderIndex)++;
    node* root = createNode(rootValue);

    if (inorderStart == inorderEnd) return root;

    int inorderIndex = search(inorder, inorderStart, inorderEnd, rootValue);

    root->left = buildTreeFromInorderPreorder(inorder, preorder, inorderStart, inorderIndex - 1, preorderIndex);
    root->right = buildTreeFromInorderPreorder(inorder, preorder, inorderIndex + 1, inorderEnd, preorderIndex);

    return root;
}

// Recursive function to construct binary tree from inorder and postorder traversals
node* buildTreeFromInorderPostorder(int inorder[], int postorder[], int inorderStart, int inorderEnd, int* postorderIndex) {
    if (inorderStart > inorderEnd) return NULL;

    int rootValue = postorder[*postorderIndex];
    (*postorderIndex)--;
    node* root = createNode(rootValue);

    if (inorderStart == inorderEnd) return root;

    int inorderIndex = search(inorder, inorderStart, inorderEnd, rootValue);

    root->right = buildTreeFromInorderPostorder(inorder, postorder, inorderIndex + 1, inorderEnd, postorderIndex);
    root->left = buildTreeFromInorderPostorder(inorder, postorder, inorderStart, inorderIndex - 1, postorderIndex);

    return root;
}

// Helper function to print inorder traversal of the tree
void inorderTraversal(node* root) {
    if (root != NULL) {
        inorderTraversal(root->left);
        printf("%d ", root->val);
        inorderTraversal(root->right);
    }
}

// Helper function to print preorder traversal of the tree
void preorderTraversal(node* root) {
    if (root != NULL) {
        printf("%d ", root->val);
        preorderTraversal(root->left);
        preorderTraversal(root->right);
    }
}

// Helper function to print postorder traversal of the tree
void postorderTraversal(node* root) {
    if (root != NULL) {
        postorderTraversal(root->left);
        postorderTraversal(root->right);
        printf("%d ", root->val);
    }
}

int main() {
    int inorder[] = {4, 2, 5, 1, 6, 3, 7};
    int preorder[] = {1, 2, 4, 5, 3, 6, 7};
    int postorder[] = {4, 5, 2, 6, 7, 3, 1};
    int size = sizeof(inorder) / sizeof(inorder[0]);

    int preorderIndex = 0;
    node* rootFromPreIn = buildTreeFromInorderPreorder(inorder, preorder, 0, size - 1, &preorderIndex);
    printf("Tree constructed from Inorder and Preorder:\nInorder Traversal: ");
    inorderTraversal(rootFromPreIn);
    printf("\n");

    int postorderIndex = size - 1;
    node* rootFromPostIn = buildTreeFromInorderPostorder(inorder, postorder, 0, size - 1, &postorderIndex);
    printf("\nTree constructed from Inorder and Postorder:\nInorder Traversal: ");
    inorderTraversal(rootFromPostIn);
    printf("\n");

    return 0;
}
