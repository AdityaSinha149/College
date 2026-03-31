/*
Bison program to evaluate arithmetic expressions with +, -, *, /
*/

%{
#include <stdio.h>
#include <stdlib.h>
%}

%token NUMBER NL
%left '+' '-'
%left '*' '/'

%%
stmt: expr NL { printf("Valid Expression\n"); exit(0); }
    ;
expr: expr '+' expr
    | expr '-' expr
    | expr '*' expr
    | expr '/' expr
    | NUMBER
    ;
%%

int yyerror(char *msg) {
    printf("Invalid Expression\n");
    exit(1);
}

int main() {
    printf("Enter arithmetic expression: ");
    yyparse();
    return 0;
}
