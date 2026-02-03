%{
#include <stdio.h>
#include <stdlib.h>
%}

%%
[0-9]+ {
            int n = atoi(yytext);
            printf("0x%X", n);
        }
%%

int yywrap() { return 1; }

int main() {
    printf("Enter input (Ctrl+D to end):\n");
    yylex();
    return 0;
}
