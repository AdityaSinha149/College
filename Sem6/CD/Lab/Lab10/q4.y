%{
#include <stdio.h>
#include <stdlib.h>

int yylex();
void yyerror(const char *s);
%}

%token NUM NL

%%

input:
      exp NL 
      { 
        printf("Valid postfix expression\n"); 
        exit(0); 
      }
      ;

exp:
      NUM
    | exp exp '+'
    | exp exp '-'
    | exp exp '*'
    | exp exp '/'
    | exp exp '^'
    ;

%%

void yyerror(const char *s) {
    printf("Invalid postfix expression\n");
    exit(1);
}

int main() {
    printf("Enter postfix expression: ");
    yyparse();
    return 0;
}