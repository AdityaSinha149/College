%{
#include <stdio.h>
#include <stdlib.h>
%}

%token INT FLOAT CHAR ID SEMI COMMA

%%
start: decl SEMI { printf("Valid declaration\n"); exit(0); }
    ;

decl: type id_list
    ;

id_list: ID
       | ID COMMA id_list
       ;

type: INT | FLOAT | CHAR;
%%

int yyerror(char *msg) {
    printf("Invalid declaration\n");
    exit(1);
}

int main() {
    printf("Enter declaration: ");
    yyparse();
    return 0;
}