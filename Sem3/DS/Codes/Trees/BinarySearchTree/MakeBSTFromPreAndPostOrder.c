#include <stdio.h>
#include <stdlib.h>

// Definition for the Binary Tree Node
struct Node {
    int val;
    struct Node* left;
    struct Node* right;
};

// Function to create a new node with a given value
struct Node* newNode(int val) {
    struct Node* node = (struct Node*)malloc(sizeof(struct Node));
    node->val = val;
    node->left = node->right = NULL;
    return node;
}

// Function to find the index of a value in the post-order sequence
int findIndex(int postOrder[], int start, int end, int value) {
    for (int i = start; i <= end; i++) {
        if (postOrder[i] == value) {
            return i;
        }
    }
    return -1;
}

// Function to construct the binary tree from pre-order and post-order
struct Node* constructTree(int preOrder[], int postOrder[], int* preIndex, int postStart, int postEnd, int size) {
    // Base case: If no elements are left to construct
    if (*preIndex >= size || postStart > postEnd) {
        return NULL;
    }

    // The root is the current pre-order element
    struct Node* root = newNode(preOrder[*preIndex]);
    (*preIndex)++;

    // If the root has no children (only one node), return it
    if (postStart == postEnd) {
        return root;
    }

    // Find the index of the left child in the post-order sequence
    int leftChildIndex = findIndex(postOrder, postStart, postEnd, preOrder[*preIndex]);

    // If a left child exists, recursively build the left subtree
    if (leftChildIndex != -1) {
        root->left = constructTree(preOrder, postOrder, preIndex, postStart, leftChildIndex, size);
    }

    // Now, recursively build the right subtree
    root->right = constructTree(preOrder, postOrder, preIndex, leftChildIndex + 1, postEnd - 1, size);

    return root;
}

// Helper function to print the tree in in-order traversal (for verification)
void inorder(struct Node* root) {
    if (root == NULL) {
        return;
    }
    inorder(root->left);
    printf("%d ", root->val);
    inorder(root->right);
}

int main() {
    // Example pre-order and post-order traversal sequences
    int preOrder[] = {1, 2, 4, 5, 3};
    int postOrder[] = {4, 5, 2, 3, 1};
    int size = sizeof(preOrder) / sizeof(preOrder[0]);

    int preIndex = 0;  // Start of pre-order
    struct Node* root = constructTree(preOrder, postOrder, &preIndex, 0, size - 1, size);

    printf("In-order traversal of the constructed tree: ");
    inorder(root);
    printf("\n");

    return 0;
}
