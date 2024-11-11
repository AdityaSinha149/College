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

// Iterative Postorder Traversal
void postorderIterative(node* root) {
    if (root == NULL) return;
    
    node* current = root;
    node* prev = NULL;
    
    push(current);
    
    while (!isEmpty()) {
        current = stack[top];
        
        // Traverse down the tree
        if (prev == NULL || prev->left == current || prev->right == current) {
            if (current->left) {
                push(current->left);
            } else if (current->right) {
                push(current->right);
            } else {
                pop();
                printf("%d -> ", current->val);
            }
        }
        // If moving up the tree
        else if (current->left == prev) {
            if (current->right) {
                push(current->right);
            } else {
                pop();
                printf("%d -> ", current->val);
            }
        } else if (current->right == prev) {
            pop();
            printf("%d -> ", current->val);
        }
        
        prev = current;
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

    return 0;
}
