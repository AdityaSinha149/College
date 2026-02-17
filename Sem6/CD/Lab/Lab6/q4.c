#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void S(void);
void L(void);
void Lp(void);
void invalid(void);
void valid(void);

int curr = 0;
char str[100];

int main(void) {
    printf("Enter  String:  ");
    if (scanf("%99s", str) != 1) {
        invalid();
    }

    S();

    if (str[curr] == '$') {
        valid();
    } else {
        invalid();
    }

    return 0;
}

void S(void) {
    if (str[curr] == '(') {
        curr++;
        L();
        if (str[curr] == ')') {
            curr++;
            return;
        }
        invalid();
    }

    if (str[curr] == 'a') {
        curr++;
        return;
    }

    invalid();
}

void L(void) {
    if (str[curr] == '(' || str[curr] == 'a') {
        S();
        Lp();
        return;
    }

    invalid();
}

void Lp(void) {
    if (str[curr] == ',') {
        curr++;
        S();
        Lp();
        return;
    }

    if (str[curr] == ')') {
        return;
    }

    invalid();
}

void invalid(void) {
    printf("-----------------ERROR!----------------\n");
    exit(0);
}

void valid(void) {
    printf("----------------SUCCESS!---------------\n");
    exit(0);
}