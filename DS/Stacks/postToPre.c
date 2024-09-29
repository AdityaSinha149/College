#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

typedef struct {
    int top;
    char** arr;
    int size;
} Stack;

void initStack(Stack* s, int size) {
    s->top = -1;
    s->size = size;
    s->arr = (char**)malloc(size * sizeof(char*));
    for (int i = 0; i < size; i++) {
        s->arr[i] = (char*)malloc(100 * sizeof(char)); // Allocate space for each string
    }
}

int isFull(Stack* s) { return s->top == s->size - 1; }
int isEmpty(Stack* s) { return s->top == -1; }

void push(Stack* s, char* str) {
    if (isFull(s)) {
        printf("Stack overflow\n");
        exit(1);
    }
    strcpy(s->arr[++s->top], str);
}

char* pop(Stack* s) {
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        exit(1);
    }
    return s->arr[s->top--];
}

int isOperator(char c) {
    return (c == '+' || c == '-' || c == '*' || c == '/');
}

void postfixToPrefix(char* postfix) {
    Stack s;
    initStack(&s, strlen(postfix));
    int len = strlen(postfix);

    for (int i = 0; i < len; i++) {
        if (isalnum(postfix[i])) {
            char op[2] = {postfix[i], '\0'};
            push(&s, op);
        } else if (isOperator(postfix[i])) {
            char operand1[100], operand2[100], result[100];
            strcpy(operand1, pop(&s));
            strcpy(operand2, pop(&s));
            // Concatenate using standard string functions
            strcpy(result, (char[]){postfix[i], '\0'}); // Create a string with the operator
            strcat(result, operand2); // Append second operand
            strcat(result, operand1); // Append first operand
            push(&s, result);
        }
    }
    printf("Prefix expression: %s\n", pop(&s));
}

int main() {
    char postfix[100];
    printf("Enter a postfix expression: ");
    scanf("%s", postfix);
    postfixToPrefix(postfix);
    return 0;
}
