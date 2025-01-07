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

// Stack for storing nodes (used for all iterative traversals)
node* stack[100];
int top = -1;

void push(node* n) {
    stack[++top] = n;
}

node* pop() {
    return stack[top--];
}

int isEmpty() {
    return top == -1;
}

// Iterative Inorder Traversal
void inorderIterative(node* root) {
    node* current = root;
    
    while (current != NULL || !isEmpty()) {
        while (current != NULL) {
            push(current);
            current = current->left;
        }
        current = pop();
        printf("%d -> ", current->val);
        
        current = current->right;
    }

    printf("NULL\n");
}

// Iterative Preorder Traversal
void preorderIterative(node* root) {
    if (root == NULL) return;
    
    push(root);
    
    while (!isEmpty()) {
        node* current = pop();
        printf("%d -> ", current->val);
        if (current->right) push(current->right);
        if (current->left) push(current->left);
    }

    printf("NULL\n");
}

typedef struct {
    node* node;
    int flag;
} Stack;

// Iterative Postorder Traversal
void postorderIterative(node* root) {
    Stack stack[20];
    int top = -1;
    node* cur = root;

    while (1) {
        while (cur != NULL) {
            stack[++top].node = cur;
            stack[top].flag = 1;
            cur = cur->left;
        }

        while (top != -1 && stack[top].flag < 0) {
            cur = stack[top--].node;
            printf("%d -> ", cur->val);
            if (top == -1){
                printf("NULL\n");
                return;
            }
        }

        if (top != -1) {
            cur = stack[top].node;
            cur = cur->right;
            stack[top].flag = -1;
        }
    }
}

void levelorderIterative(node* root) {
    if (root == NULL) return;

    node* queue[100];
    int front = 0, rear = 0;
    queue[rear++] = root;

    while (front < rear) {
        node* current = queue[front++];
        printf("%d -> ", current->val);

        if (current->left) queue[rear++] = current->left;
        if (current->right) queue[rear++] = current->right;
    }

    printf("NULL\n");
}

int main() {
    // Constructing a binary tree
    node* root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);

    // Iterative Traversals
    printf("Inorder traversal (Iterative): ");
    inorderIterative(root);
    printf("\n");

    printf("Preorder traversal (Iterative): ");
    preorderIterative(root);
    printf("\n");

    printf("Postorder traversal (Iterative): ");
    postorderIterative(root);
    printf("\n");

    printf("Levelorder traversal (Iterative): ");
    levelorderIterative(root);
    printf("\n");

    return 0;
}
