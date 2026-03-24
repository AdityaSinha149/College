#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX 100

int stack[MAX];
int top = -1;

char input[MAX];
int ip = 0;


int action[12][6] = {
    { 5,   0,   0,   4,   0,   0 },
    { 0,   6,   0,   0,   0, 999 },
    { 0,  -2,   7,   0,  -2,  -2 },
    { 0,  -4,  -4,   0,  -4,  -4 },
    { 5,   0,   0,   4,   0,   0 },
    { 0,  -6,  -6,   0,  -6,  -6 },
    { 5,   0,   0,   4,   0,   0 },
    { 5,   0,   0,   4,   0,   0 },
    { 0,  -1,   7,   0,  -1,  -1 },
    { 0,  -3,  -3,   0,  -3,  -3 },
    { 0,   6,   0,   0,  11,   0 },
    { 0,  -5,  -5,   0,  -5,  -5 } 
};

int go_to[12][3] = {
    { 1, 2, 3 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    {10, 2, 3 },
    { 0, 0, 0 },
    { 0, 8, 3 },
    { 0, 0, 9 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 },
    { 0, 0, 0 }
};

int prod_len[] = {0, 3, 1, 3, 1, 3, 1};
char prod_lhs[] = {' ', 'E', 'E', 'T', 'T', 'F', 'F'};

int get_col(char c) {
    switch(c) {
        case 'i': return 0;
        case '+': return 1;
        case '*': return 2;
        case '(': return 3;
        case ')': return 4;
        case '$': return 5;
        default: return -1;
    }
}

int get_nt_col(char c) {
    switch(c) {
        case 'E': return 0;
        case 'T': return 1;
        case 'F': return 2;
        default: return -1;
    }
}

void push(int x) {
    stack[++top] = x;
}

int pop() {
    return stack[top--];
}

void parse() {
    push(0);

    while(1) {
        int state = stack[top];
        int col = get_col(input[ip]);
        int act = action[state][col];

        if (act == 999) {
            printf("ACCEPTED\n");
            return;
        }
        else if (act > 0) {
            push(col); 
            push(act);
            ip++;
        }
        else if (act < 0) {
            int prod = -act;
            int len = prod_len[prod];

            for(int i = 0; i < 2*len; i++)
                pop();

            int state2 = stack[top];
            int nt_col = get_nt_col(prod_lhs[prod]);
            push(nt_col);
            push(go_to[state2][nt_col]);
        }
        else {
            printf("ERROR\n");
            return;
        }
    }
}

int main() {
    printf("Enter input (use i for id): ");
    scanf("%s", input);
    strcat(input, "$");

    parse();
    return 0;
}