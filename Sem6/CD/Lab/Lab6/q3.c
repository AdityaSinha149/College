#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void S();
void A();
void Ap();
void B();

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
    if(str[curr] == 'a') {
        curr++;
        A();
        if(str[curr] == 'c') curr++;
        B();
        if(str[curr] == 'e') curr++;
        return;
    }
    invalid();
}

void A() {
    if(str[curr] == 'b') {
        curr++;
        Ap();
        return;
    }
    invalid();
}

void Ap() {
    if(str[curr] == 'b') {
        curr++;
        Ap();
        return;
    }
    else if(str[curr] == 'c')
        return;
    invalid();
}

void B() {
    if(str[curr] == 'd') {
        curr++;
        return;
    }
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