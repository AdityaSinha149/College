#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct node {
    char val;
    struct node* left;
    struct node* right;
} node;

node* createNode(char value) {
    node* newnode = (node*)malloc(sizeof(node));
    newnode->val = value;
    newnode->left = newnode->right = NULL;
    return newnode;
}

// Stack for storing nodes
node* stack[100];
int top = -1;

void push(node* n) {
    stack[++top] = n;
}

node* pop() {
    return stack[top--];
}

// Function to create a tree from a prefix expression
node* createTreeFromPrefix(char* expr) {
    for (int i = strlen(expr) - 1; i >= 0; i--) {
        char c = expr[i];
        
        if (isalnum(c))
            push(createNode(c));
        else {
            node* newNode = createNode(c);
            newNode->left = pop();
            newNode->right = pop();
            push(newNode);
        }
    }
    return pop();
}

// Function to create a tree from a postfix expression
node* createTreeFromPostfix(char* expr) {
    for (int i = 0; i < strlen(expr); i++) {
        char c = expr[i];
        
        if (isalnum(c)) 
            push(createNode(c));
        else {
            node* newNode = createNode(c);
            newNode->right = pop();
            newNode->left = pop();
            push(newNode);
        }
    }
    return pop();
}

// Tree Traversal Functions
void inorder(node* root) {
    if (root) {
        inorder(root->left);
        printf("%c ", root->val);
        inorder(root->right);
    }
}

void postorder(node* root) {
    if (root) {
        postorder(root->left);
        postorder(root->right);
        printf("%c ", root->val);
    }
}

void preorder(node* root) {
    if (root) {
        printf("%c ", root->val);
        preorder(root->left);
        preorder(root->right);
    }
}

int main() {
    char prefixExp[] = "*+AB-CD";
    char postfixExp[] = "AB+C*CD-";

    // Construct the trees from the prefix and postfix expressions
    node* rootPrefix = createTreeFromPrefix(prefixExp);
    node* rootPostfix = createTreeFromPostfix(postfixExp);
    
    // Display traversals for Prefix expression tree
    printf("Prefix expression tree:\n");
    printf("Inorder traversal: ");
    inorder(rootPrefix);
    printf("\n");

    printf("Preorder traversal: ");
    preorder(rootPrefix);
    printf("\n");

    printf("Postorder traversal: ");
    postorder(rootPrefix);
    printf("\n");

    // Display traversals for Postfix expression tree
    printf("\nPostfix expression tree:\n");
    printf("Inorder traversal: ");
    inorder(rootPostfix);
    printf("\n");

    printf("Preorder traversal: ");
    preorder(rootPostfix);
    printf("\n");

    printf("Postorder traversal: ");
    postorder(rootPostfix);
    printf("\n");

    return 0;
}
