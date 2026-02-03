%{
#include <stdio.h>

int pos_int = 0;
int neg_int = 0;
int pos_float = 0;
int neg_float = 0;
%}

%%
-[0-9]+\.[0-9]+     { neg_float++; }
[0-9]+\.[0-9]+      { pos_float++; }
-[1-9][0-9]*        { neg_int++; }
[1-9][0-9]*         { pos_int++; }
[ \t\n]+            { }
.                   { }
%%

int yywrap() { return 1; }

int main() {
    printf("Enter numbers (Ctrl+D to end):\n");
    yylex();

    printf("Positive Integers        = %d\n", pos_int);
    printf("Negative Integers        = %d\n", neg_int);
    printf("Positive Floating Point  = %d\n", pos_float);
    printf("Negative Floating Point  = %d\n", neg_float);

    return 0;
}
