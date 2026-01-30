#ifndef STACK_H
#define STACK_H

#define STACK_SIZE 100
#include "lexAnalyszer.h"

typedef struct stack{
    char arr[STACK_SIZE];
    int top;
}stack;

// prototypes
static void pushStack(stack *s, char c);
static char popStack(stack *s);
static char peekStack(stack *s);
static int isEmpty(stack *s);

static void updateStack(stack *scopeStack, token currToken) {
    char symbol = currToken.tokenName[0];

    if (symbol == '{' || symbol == '(' || symbol == '[') {
        pushStack(scopeStack, symbol);
    }
    else if (symbol == '}' || symbol == ')' || symbol == ']') {
        char top = peekStack(scopeStack);

        if ((symbol == '}' && top == '{') ||
            (symbol == ')' && top == '(') ||
            (symbol == ']' && top == '[')) {
            popStack(scopeStack);
        }
    }
}


static void pushStack(stack *s, char c) {
    if (s->top == STACK_SIZE - 1)
        return;

    s->arr[++(s->top)] = c;
}

static char popStack(stack *s) {
    if (s->top == -1)
        return '\0';

    return s->arr[(s->top)--];
}

static char peekStack(stack *s) {
    if (s->top == -1)
        return '\0';

    return s->arr[s->top];
}

static int isEmpty(stack *s) {
    return s->top == -1;
}

#endif
