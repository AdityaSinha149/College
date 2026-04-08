%{
#include <stdio.h>
#include <stdlib.h>

int yylex();
void yyerror(const char *s);
%}

%union {
    int num;
}

%token IF ELSE ID RELOP ASSIGN SEMI LPAREN RPAREN
%token <num> NUMBER

%left '+' '-'
%left '*' '/'
%nonassoc IFX
%nonassoc ELSE

%%

start: ifstmt SEMI 
      { 
        printf("Valid decision statement\n"); 
        exit(0); 
      }
     ;

ifstmt:
      IF LPAREN cond RPAREN assign_stmt %prec IFX
    | IF LPAREN cond RPAREN assign_stmt ELSE assign_stmt
    ;

cond: expr RELOP expr ;

expr: expr '+' term
    | expr '-' term
    | term
    ;

term: term '*' factor
    | term '/' factor
    | factor
    ;

factor:
      ID
    | NUMBER
    ;

assign_stmt: ID ASSIGN expr ;

%%

void yyerror(const char *s) {
    printf("Invalid decision statement\n");
    exit(1);
}

int main() {
    printf("Enter decision statement: ");
    yyparse();
    return 0;
}