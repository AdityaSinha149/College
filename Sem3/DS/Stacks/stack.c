#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
#include <stdbool.h>

typedef struct {
    int top;
    char *arr;
    int size;
} stack;

void stacki(stack* s, int size) {
    s->top = -1;
    s->size = size;
    s->arr = (char*)malloc(size * sizeof(char));
}

bool isEmpty(stack *s) {
    return s->top == -1;
}

bool isFull(stack *s) {
    return s->top == s->size - 1;
}

void push(stack* s, char elem) {
    if (isFull(s)) {
        printf("Stack overflow\n");
        exit(1);
    }
    s->arr[++s->top] = elem;
}

char pop(stack* s) {
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        exit(1);
    }
    return s->arr[s->top--];
}

char peek(stack* s) {
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        exit(1);
    }
    return s->arr[s->top];
}

int main() {
    char str[] = "HELLO";  // String to be reversed
    int len = strlen(str);
    
    // Initialize the stack
    stack s;
    stacki(&s, len);

    // Push each character of the string onto the stack
    for (int i = 0; i < len; i++) {
        push(&s, str[i]);
    }

    printf("Reversed string: ");
    
    // Pop characters from the stack to reverse the string
    while (!isEmpty(&s)) {
        printf("%c", pop(&s));
    }
    printf("\n");

    // Free the dynamically allocated memory for the stack
    free(s.arr);

    return 0;
}
