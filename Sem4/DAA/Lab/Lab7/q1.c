#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int data;
    struct node *left, *right;
    int height;
}   TreeNode;

TreeNode* createNode(int data){
    TreeNode* newNode = (TreeNode*)malloc(sizeof(TreeNode));
    newNode->data = data;
    newNode->left = NULL;
    newNode->right = NULL;
    newNode->height = 1;
    return newNode;
}

int nodeHeight(TreeNode* node) {
    return node ? node->height : 0;
}

int max(int a, int b) {
    return (a > b)? a : b;
}

TreeNode* R(TreeNode* y) {
    TreeNode* x = y->left;
    TreeNode* T2 = x->right;

    x->right = y;
    y->left = T2;

    y->height = max(nodeHeight(y->left), nodeHeight(y->right)) + 1;
    x->height = max(nodeHeight(x->left), nodeHeight(x->right)) + 1;

    return x;
}

TreeNode* L(TreeNode* x) {
    TreeNode* y = x->right;
    TreeNode* T2 = y->left;

    y->left = x;
    x->right = T2;

    x->height = max(nodeHeight(x->left), nodeHeight(x->right)) + 1;
    y->height = max(nodeHeight(y->left), nodeHeight(y->right)) + 1;

    return y;
}

TreeNode* LR(TreeNode* root) {
    root->left = L(root->left);
    return R(root);
}

TreeNode* RL(TreeNode* root) {
    root->right = R(root->right);
    return L(root);
}

TreeNode* insert(TreeNode* root, int data){
    if(root == NULL)
        return createNode(data);
    if(data < root->data)
        root->left = insert(root->left, data);
    else if(data > root->data)
        root->right = insert(root->right, data);
    else
        return root;

    root->height = 1 + max(nodeHeight(root->left), nodeHeight(root->right));

    int balance = nodeHeight(root->left) - nodeHeight(root->right);

    if(balance > 1 && data < root->left->data)
        return R(root);
    if(balance < -1 && data > root->right->data)
        return L(root);
    if(balance > 1 && data > root->left->data)
        return LR(root);
    if(balance < -1 && data < root->right->data)
        return RL(root);
    
    return root;
}

void printTree(TreeNode* root, int space) {
    if (root == NULL)
        return;

    space += 10;

    printTree(root->right, space);

    printf("\n");
    for (int i = 10; i < space; i++)
        printf(" ");
    printf("%d\n", root->data);

    printTree(root->left, space);
}

int main() {
    TreeNode *root = NULL;
    int n, data;

    printf("Enter the number of nodes: ");
    scanf("%d", &n);

    for (int i = 0; i < n; i++) {
        printf("Enter node value: ");
        scanf("%d", &data);
        root = insert(root, data);
    }

    printf("AVL Tree:\n");
    printTree(root, 0);

    return 0;
}

