#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

void Program();
void declarations();
void data_type();
void identifier_list();
void assign_stat();
void invalid(char*);
void valid();

int curr = 0;
char str[1000];

void main() {
    printf("Enter String: ");
    scanf("%s", str);
    Program();
    if(str[curr] == '$') valid();
    else {
        printf("Unexpected character: %c at position %d\n", str[curr], curr);
        invalid("Extra characters after valid program");
    }
}

void Program() {
    if(strncmp(&str[curr], "main", 4) == 0) {
        curr += 4;
        if(str[curr] == '(') {
            curr++;
            if(str[curr] == ')') {
                curr++;
                if(str[curr] == '{') {
                    curr++;
                    declarations();
                    assign_stat();
                    if(str[curr] == '}') {
                        curr++;
                        return;
                    }
                    invalid("Expected '}' to close program");
                }
                invalid("Expected '{' after main()");
            }
            invalid("Expected ')' after '('");
        }
        invalid("Expected '(' after main");
    }
    invalid("Expected 'main' keyword");
}

void declarations() {
    if(strncmp(&str[curr], "int", 3) == 0 || strncmp(&str[curr], "char", 4) == 0) {
        data_type();
        identifier_list();
        if(str[curr] == ';') {
            curr++;
            declarations();
            return;
        }
        invalid("Expected ';' after declaration");
    }
    return;
}

void data_type() {
    if(strncmp(&str[curr], "int", 3) == 0) {
        curr += 3;
        return;
    }
    if(strncmp(&str[curr], "char", 4) == 0) {
        curr += 4;
        return;
    }
    invalid("Expected 'int' or 'char' data type");
}

void identifier_list() {
    if(isalpha(str[curr]) || str[curr] == '_') {
        while(isalnum(str[curr]) || str[curr] == '_') {
            curr++;
        }
        if(str[curr] == ',') {
            curr++;
            identifier_list();
            return;
        }
        return;
    }
    invalid("Expected identifier");
}

void assign_stat() {
    if(isalpha(str[curr]) || str[curr] == '_') {
        while(isalnum(str[curr]) || str[curr] == '_') {
            curr++;
        }
        if(str[curr] == '=') {
            curr++;
            if(isalpha(str[curr]) || str[curr] == '_') {
                while(isalnum(str[curr]) || str[curr] == '_') {
                    curr++;
                }
            }
            else if(isdigit(str[curr])) {
                while(isdigit(str[curr])) {
                    curr++;
                }
            }
            else {
                invalid("Expected identifier or number after '='");
            }
            
            if(str[curr] == ';') {
                curr++;
                return;
            }
            invalid("Expected ';' after assignment");
        }
        invalid("Expected '=' in assignment statement");
    }
    invalid("Expected identifier in assignment statement");
}

void invalid(char* msg) {
    printf("ERROR!\n");
    printf("Error at position %d: %s\n", curr, msg);
    printf("Remaining string: %s\n", &str[curr]);
    exit(0);
}

void valid() {
    printf("SUCCESS!\n");
    printf("Valid program structure!\n");
    exit(0);
}
