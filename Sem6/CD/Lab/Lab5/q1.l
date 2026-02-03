%{
int vowels = 0, consonants = 0;
%}

%%
[aeiouAEIOU] { vowels++; }
[a-zA-Z] { consonants++; }
. { }
%%

int yywrap() { return 1; }

int main() {
    char stat[100];

    printf("Enter the string: ");
    scanf("%s", stat);

    yy_scan_string(stat);
    yylex();

    printf("Vowels = %d\n", vowels);
    printf("Consonants = %d\n", consonants);

    return 0;
}
