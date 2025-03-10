#include <stdio.h>
#include <stdlib.h>

typedef struct {
    char data;
    struct node *next;
}node;

node createnode(node* newnode,char key){
    newnode=(node*)malloc(sizeof(node));
    newnode->data=key;
    newnode->next=NULL;
    return newnode;
}

void insert(node* hashmap[],int m,char key){
    int index=key%m;
    node* newnode=createnode(newnode,key);    

    if(hashmap[index]==NULL)
        hashmap[index]=newnode;
    else{
        node* temp=hashmap[index];
        while(temp->next!=NULL)
            temp=temp->next;
        temp->next=newnode;
    }
}

int search(node* hashmap[],int m,char key){
    int index=key%m;
    node* temp=hashmap[index];
    while(temp!=NULL){
        if(temp->data==key)
            return 1;
        temp=temp->next;
    }
    return 0;
}

insert elements in while(1) untill -1
int main(){
    int m;
    printf("Enter the size of the hash table: ");
    scanf("%d",&m);
    node* hashmap[m];
    for(int i=0;i<m;i++)
        hashmap[i]=NULL;

    while(1){
        int key;
        printf("Enter the element to insert(-1 to stop): ");
        scanf("%d",&key);
        if(key==-1)
            break;
        insert(hashmap,m,key);
    }

    int key;
    printf("Enter the element to search(-1 to stop): ");
    scanf("%d",&key);
    if(key==-1)
        break;
    if(search(hashmap,m,key))
        printf("Element found\n");
    else
        printf("Element not found\n");
    return 0;
}