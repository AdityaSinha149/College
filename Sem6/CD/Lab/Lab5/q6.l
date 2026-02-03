%{
#include <ctype.h>
FILE *out;
%}

%x SINGLE
%x MULTI

%%
"//"            { fprintf(out, "%s", yytext); BEGIN(SINGLE); }
"/*"            { fprintf(out, "%s", yytext); BEGIN(MULTI); }

<SINGLE>\n      { fprintf(out, "%s", yytext); BEGIN(INITIAL); }
<SINGLE>.       { fprintf(out, "%s", yytext); }

<MULTI>"*/"     { fprintf(out, "%s", yytext); BEGIN(INITIAL); }
<MULTI>.|\n     { fprintf(out, "%s", yytext); }

[A-Z]           { fprintf(out, "%c", tolower(yytext[0])); }
.|\n            { fprintf(out, "%s", yytext); }
%%

int yywrap() { return 1; }

int main(int argc, char *argv[]) {
    if (argc != 2) return 1;

    yyin = fopen(argv[1], "r");
    if (!yyin) return 1;

    out = fopen("ans.txt", "w");
    if (!out) return 1;

    yylex();

    fclose(yyin);
    fclose(out);
    return 0;
}
