#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

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

bool isEqual(node* root1, node* root2){
    if(root1==NULL&&root2==NULL)return true;
    if(root1==NULL||root2==NULL)return false;
    return (root1->val==root2->val)&&isEqual(root1->left,root2->left)&&isEqual(root1->right,root2->right);
}

node* copyBT(node* root){
    if(root==NULL)return NULL;
    node* newBT=createNode(root->val);
    newBT->left=copyBT(root->left);
    newBT->right=copyBT(root->right);
    return newBT;
}

int main() {
    node* root = createNode(1);
    root->left = createNode(2);
    root->right = createNode(3);
    root->left->left = createNode(4);
    root->left->right = createNode(5);

    node* root2=copyBT(root);
    if(isEqual(root,root2))
        printf("Both trees are equal.\n");
    else
        printf("Both trees are not equal.\n");

    return 0;
}
