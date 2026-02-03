%{
int scanf_count = 0;
int printf_count = 0;

FILE *out;
%}

%%
"scanf"     { fprintf(out, "READ");  scanf_count++; }
"printf"    { fprintf(out, "WRITE"); printf_count++; }
.|\n        { fprintf(out, "%s", yytext); }
%%

int yywrap() { return 1; }

int main(int argc, char *argv[]) {

    if (argc != 2) {
        printf("Usage: %s <input_file.c>\n", argv[0]);
        return 1;
    }

    yyin = fopen(argv[1], "r");
    if (!yyin) {
        perror("Input file error");
        return 1;
    }

    out = fopen("ans.txt", "w");
    if (!out) {
        perror("Output file error");
        return 1;
    }

    yylex();

    printf("\n\nNumber of scanf  = %d\n", scanf_count);
    printf("Number of printf = %d\n", printf_count);

    fclose(yyin);
    fclose(out);
    return 0;
}
