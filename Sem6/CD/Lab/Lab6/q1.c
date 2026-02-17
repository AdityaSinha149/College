#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void S();
void T();
void Tp();
void invalid();
void valid();


int curr = 0;
char str[100];

void main() {
    printf("Enter  String:  ");  
    scanf("%s", str);  
    S();
    if(str[curr] == '$') valid();
    else {
        printf("%c\n", str[curr]);
        invalid();
    }  
}
void S() {
    if(str[curr] == 'a' || str[curr] == '>') {
        curr++;
        return;
    }
    if(str[curr] == '(') {
        curr++;
        T();
        if(str[curr] == ')') {
            curr++;
            return;
        }
        invalid();
    }
    invalid();
}

void T() {
    if(str[curr] == 'a' || str[curr] == '>' || str[curr] == '(') {
        S();
        Tp();
        return;
    }
    invalid();
}

void Tp() {
    if(str[curr] == ',') {
        curr++;
        S();
        Tp();
        return;
    }
    if(str[curr] == ')') return;
    invalid();
}

void invalid() {  
   printf("-----------------ERROR!----------------\n");  
   exit(0);  
}

void valid() {  
  printf("----------------SUCCESS!---------------\n");    
 exit(0);  
}