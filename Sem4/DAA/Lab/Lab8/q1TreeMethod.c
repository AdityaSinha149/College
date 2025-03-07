#include <stdio.h>
#include <stdlib.h>

struct TreeNode {
    int value;
    struct TreeNode *left;
    struct TreeNode *right;
};

struct TreeNode* createNode(int value) {
    struct TreeNode *newNode = (struct TreeNode *)malloc(sizeof(struct TreeNode));
    newNode->value = value;
    newNode->left = newNode->right = NULL;
    return newNode;
}

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

void heapify(struct TreeNode* root) {
    if (root == NULL) {
        return;
    }

    struct TreeNode *largest = root;
    if (root->left != NULL && root->left->value > largest->value) {
        largest = root->left;
    }

    if (root->right != NULL && root->right->value > largest->value) {
        largest = root->right;
    }

    if (largest != root) {
        swap(&root->value, &largest->value);
        heapify(largest);
    }
}

void buildTree(struct TreeNode** root, int arr[], int n, int index) {
    if (index >= n) {
        return;
    }

    *root = createNode(arr[index]);
    buildTree(&((*root)->left), arr, n, 2 * index + 1);
    buildTree(&((*root)->right), arr, n, 2 * index + 2);
}

void printTree(struct TreeNode *root) {
    if (root == NULL) {
        return;
    }

    printf("%d ", root->value);
    printTree(root->left);
    printTree(root->right);
}

void heapifyTree(struct TreeNode *root) {
    if (root == NULL) {
        return;
    }

    heapify(root);
    heapifyTree(root->left);
    heapifyTree(root->right);
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    struct TreeNode *root = NULL;
    buildTree(&root, arr, n, 0);

    printf("Original Tree (Preorder traversal): ");
    printTree(root);
    printf("\n");

    heapifyTree(root);

    printf("Heapified Tree (Preorder traversal): ");
    printTree(root);
    printf("\n");

    return 0;
}
