%{
#include <stdio.h>
int words = 0, chars = 0, blanks = 0, lines = 0;
%}

%%
[a-zA-Z]+        { words++; chars += yyleng; }
[ \t\r\f\v]+     { blanks += yyleng; chars += yyleng; }
\n               { lines++; chars++; }
\"               { return 0; }
.                { chars++; }
%%

int yywrap() { return 1; }

int main() {
    printf("Enter input (you can type multiple lines, end with \"):\n");
    yylex();

    printf("\nWords      = %d\n", words);
    printf("Characters = %d\n", chars);
    printf("Blanks     = %d\n", blanks);
    printf("Lines      = %d\n", lines);

    return 0;
}
