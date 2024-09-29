#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <stdbool.h>

typedef struct {
    int top;
    int *arr;
    int size;
} stack;

void stacki(stack *s, int size) {
    s->top = -1;
    s->size = size;
    s->arr = (int *)malloc(size * sizeof(int));
}

bool isEmpty(stack *s) {
    return s->top == -1;
}

bool isFull(stack *s) {
    return s->top == s->size - 1;
}

void push(stack *s, int elem) {
    if (isFull(s)) {
        printf("Stack overflow\n");
        exit(1);
    }
    s->arr[++s->top] = elem;
}

int pop(stack *s) {
    if (isEmpty(s)) {
        printf("Stack underflow\n");
        exit(1);
    }
    return s->arr[s->top--];
}

int evaluatePostfix(const char *postfix) {
    stack s;
    int len = strlen(postfix);
    stacki(&s, len);

    for (int i = 0; i < len; i++) {
        if (isdigit(postfix[i])) {
            push(&s, postfix[i] - '0');
        } else if (postfix[i] == ' ') {
            continue;
        } else {
            int val2 = pop(&s);
            int val1 = pop(&s);
            switch (postfix[i]) {
                case '+': push(&s, val1 + val2); break;
                case '-': push(&s, val1 - val2); break;
                case '*': push(&s, val1 * val2); break;
                case '/':
                    if (val2 != 0) {
                        push(&s, val1 / val2);
                    } else {
                        printf("Division by zero error\n");
                        free(s.arr);
                        exit(1);
                    }
                    break;
                default:
                    printf("Unknown operator %c\n", postfix[i]);
                    free(s.arr);
                    exit(1);
            }
        }
    }

    int result = pop(&s);
    free(s.arr);
    return result;
}

int main() {
    printf("Enter postfix expression: ");
    char postfix[1000];
    if (fgets(postfix, sizeof(postfix), stdin) != NULL) {
        postfix[strcspn(postfix, "\n")] = '\0';

        int result = evaluatePostfix(postfix);
        printf("Result: %d\n", result);
    } else {
        printf("Error reading input.\n");
    }
    return 0;
}
