#include <stdio.h>
#include <stdlib.h>

typedef struct node{
    int val;
    struct node* left;
    struct node* right;
}node;

node* createNode(){
    node* newnode=(node*)malloc(sizeof(node));
    newnode->left=newnode->right=NULL;
    return newnode;
}

node* createTree(int x){
    if(x!=-1){
        node* temp=createNode();
        temp->val=x;
        
        int y;
        printf("enter the left chilld of %d : ",x);
        scanf("%d",&y);
        temp->left=createTree(y);
        
        printf("enter the right chilld of %d : ",x);
        scanf("%d",&y);
        temp->right=createTree(y);
        
        return temp;
    }
    return NULL;
}

void display(node* root){
    if (root != NULL) {
        display(root->left);
        printf("%d -> ", root->val);
        display(root->right);
    }
}

int main() {
    printf("enter the value of root node : ");
    int x;
    scanf("%d",&x);
    node* root=createTree(x);
    display(root);
    return 0;
}