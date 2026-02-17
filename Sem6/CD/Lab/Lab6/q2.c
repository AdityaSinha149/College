#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void S();
void U();
void V();
void W();

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

void S(){
    if(str[curr] == 'a' || str[curr] == 'd') {
        U();
        V();
        W();
        return;
    }
    invalid();
}

void U(){
    if(str[curr] == 'a') {
        curr++;
        S();
        if(str[curr] == 'b') curr++;
        return;
    }
    if(str[curr] == 'd') {
        curr++;
        return;
    }
    if(str[curr] == '(') {
        curr++;
        S();
        if(str[curr] == ')') curr++;
        return;
    }
    invalid();
}
void V(){
    if(str[curr] == 'a') {
        curr++;
        V();
        return;
    }
    else if(str[curr] == 'b' || str[curr] == 'c' || str[curr] == ')' || str[curr] == '$')
        return;
    invalid();
}
void W(){
    if(str[curr] == 'c') {
        curr++;
        W();
        return;
    }
    else if(str[curr] == 'b' || str[curr] == ')' || str[curr] == '$')
        return;
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