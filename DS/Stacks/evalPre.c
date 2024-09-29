#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>
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

int evaluatePrefix(const char *prefix) {
    stack s;
    int len = strlen(prefix);
    stacki(&s, len);

    for (int i = len - 1; i >= 0; i--) {
        if (isdigit(prefix[i])) {
            push(&s, prefix[i] - '0');
        } else if (prefix[i] == ' ') {
            continue;
        } else {
            int val1 = pop(&s);
            int val2 = pop(&s);
            switch (prefix[i]) {
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
                    printf("Unknown operator %c\n", prefix[i]);
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
    printf("Enter prefix expression: ");
    char prefix[1000];
    if (fgets(prefix, sizeof(prefix), stdin) != NULL) {
        prefix[strcspn(prefix, "\n")] = '\0';

        int result = evaluatePrefix(prefix);
        printf("Result: %d\n", result);
    } else {
        printf("Error reading input.\n");
    }
    return 0;
}
