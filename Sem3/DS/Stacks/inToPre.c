// Online C compiler to run C program online
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>

typedef struct{
    int top;
    char *arr;
    int size;
}stack;

void stacki(stack* s,int size){
    s->top=-1;
    s->size=size;
    s->arr=(char*)malloc(size*sizeof(char));
}

bool isEmpty(stack *s){
    return s->top==-1;
}

bool isFull(stack *s){
    return s->top==s->size-1;
}

void push(stack* s,char elem){
    if (isFull(s)) {
        printf("Stack overflow\n");
        exit(1);
    }
    s->arr[++s->top]=elem;
}

char pop(stack* s){
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        exit(1);
    }
    return s->arr[s->top--];
}

char peek(stack* s){
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        exit(1);
    }
    return s->arr[s->top];
}

int priority(char ch) {
    switch (ch) {
        case '^': return 3;
        case '*':
        case '/': return 2;
        case '+':
        case '-': return 1;
        default: return 0;
    }
}

void reverse(char *str) {
    int n = strlen(str);
    for (int i = 0; i < n / 2; i++) {
        // Swap parentheses
        if (str[i] == '(') str[i] = ')';
        else if (str[i] == ')') str[i] = '(';
        
        if (str[n - 1 - i] == '(') str[n - 1 - i] = ')';
        else if (str[n - 1 - i] == ')') str[n - 1 - i] = '(';

        // Swap characters
        char temp = str[i];
        str[i] = str[n - 1 - i];
        str[n - 1 - i] = temp;
    }

    // Handle the middle character for odd-length strings
    if (n % 2 == 1) {
        if (str[n / 2] == '(') str[n / 2] = ')';
        else if (str[n / 2] == ')') str[n / 2] = '(';
    }
}

char* inToPost(char* infix){
    int len=strlen(infix);
    stack s;
    stacki(&s,len);
    char* post=(char*)malloc((len+1)*sizeof(char));
    int j=0;
    for(int i=0;i<len;i++){
        if(isalnum(infix[i]))
            post[j++]=infix[i];
        else if(infix[i]=='(')
            push(&s,'(');
        else if(infix[i]==')'){
            while(!isEmpty(&s)&&peek(&s)!='(')
                post[j++]=pop(&s);
            if (isEmpty(&s) || peek(&s) != '(') {
                printf("Invalid expression\n");
                free(post);
                exit(1);
            }
            pop(&s);
        }
        else{
            while(!isEmpty(&s) && priority(infix[i])<priority(peek(&s)))
                post[j++]=pop(&s);
            push(&s,infix[i]);
        }
    }
    while(!isEmpty(&s))
        post[j++]=pop(&s);
    post[j]='\0';
    return post;
}

int main() {
    printf("enter equation:");
    char eq[1000];
    gets(eq);
    reverse(eq);
    char* post=inToPost(eq);
    reverse(post);
    puts(post);
    
}