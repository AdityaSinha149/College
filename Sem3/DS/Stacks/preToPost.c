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

void prefixToPostfix(char* prefix) {
    Stack s;
    initStack(&s, strlen(prefix));
    int len = strlen(prefix);

    // Traverse the prefix expression from right to left
    for (int i = len - 1; i >= 0; i--) {
        if (isalnum(prefix[i])) {
            char op[2] = {prefix[i], '\0'};
            push(&s, op);
        } else if (isOperator(prefix[i])) {
            char operand1[100], operand2[100], result[100];
            strcpy(operand1, pop(&s));
            strcpy(operand2, pop(&s));
            // Concatenate to form the postfix expression
            strcpy(result, operand1);  // First operand
            strcat(result, operand2);  // Second operand

            char operator[2] = {prefix[i], '\0'}; // Create a string for the operator
            strcat(result, operator); // Append the operator

            push(&s, result);
        }
    }
    printf("Postfix expression: %s\n", pop(&s));
}

int main() {
    char prefix[100];
    printf("Enter a prefix expression: ");
    scanf("%s", prefix);
    prefixToPostfix(prefix);
    return 0;
}
