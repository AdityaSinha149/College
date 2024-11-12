#include <stdio.h>
#include <stdlib.h>

typedef struct node {
    int val;
    struct node* prev;
    struct node* next;
} node;

node* createNode(int x) {
    node* newnode = (node*)malloc(sizeof(node));
    newnode->val = x;
    newnode->prev = newnode->next = newnode;
    return newnode;
}

node* insertAtBeg(node* head, int n){
    node* newnode=createNode(n);
    if(head==NULL)
        return newnode;
    newnode->prev=head->prev;
    head->prev->next=newnode;
    newnode->next=head;
    head->prev=newnode;
    head=newnode;
    return head;
}

node* insertAtEnd(node* head, int n){
    node* newnode=createNode(n);
    if(head==NULL)
        return newnode;
    node* temp=head->prev;
    temp->next=newnode;
    newnode->prev=temp;
    newnode->next=head;
    head->prev=newnode;
    return head;
}

node* insertAtMiddle(node* head, int x, int n){
    if(n==1)
        return insertAtBeg(head, x);
    node* temp=head;
    for(int i=1;i<n;i++){
        if(temp->next==head){
            printf("Position not found\n");
            return head;
        }
        temp=temp->next;
    }
    if(temp->next==head)
        return insertAtEnd(head, x);
    node* newnode=createNode(x);
    newnode->next=temp->next;
    newnode->prev=temp;
    temp->next->prev=newnode;
    temp->next=newnode;
    return head;
}

void display(node* head) {
    if (head == NULL) {
        printf("List is empty\n");
        return;
    }
    node* temp = head;
    do {
        printf("%d -> ", temp->val);
        temp = temp->next;
    } while (temp != head);  // Stop when we've looped back to the head
    printf("(head)\n");
}

int main() {
    node* head = NULL;

    // Insert some elements
    head = insertAtEnd(head, 10);
    head = insertAtEnd(head, 20);
    head = insertAtBeg(head, 5);
    head = insertAtMiddle(head, 15, 3);

    printf("Circular Doubly Linked List:\n");
    display(head);

    return 0;
}
