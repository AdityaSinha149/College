#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

typedef struct token{
    char token_name [50];
    unsigned int row,col;
}token;

typedef struct {
    char name[100];
    char value[100];
} Macro;

Macro macros[100];
int macroCount = 0;

const char *keywords[] = {
    "auto", "break", "case", "char", "const",
    "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for",
    "goto", "if", "int", "long", "register",
    "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while"
};

//Preprocessing part
void addMacro(const char *name, const char *value);
const char* getMacroValue(const char *name);
void preprocess(FILE *src, FILE * dst);

//Token identifying
token isKeyword(char ch, FILE *src, int row, int col);
token isIdentifier(char ch, FILE *src, int row, int col);

token isRelationalOperator(char ch, FILE *src, int row, int col);
token isArithmeticOperator(char ch, FILE *src, int row, int col);
token isLogicalOperator(char ch, FILE *src, int row, int col);

token isOperator(char ch, FILE *src, int row, int col);

token isSymbol(char ch, FILE *src, int row, int col);

token isStringLiteral(char ch, FILE *src, int row, int col);

token isNumber(char ch, FILE *src, int row, int col);

//Token Server
token getNextToken(FILE *src, int *row, int *col);

//Helpers
void PrintToken(token t, FILE *src, FILE *dst);

int main(){
    printf ("Enter program to lexical analyse: ");
    char file[100];
    scanf ("%s", file);

    FILE *src = fopen(file, "r");
    if (src == NULL) {
        perror("error ");
        return 1;
    }

    FILE *tmp = fopen("tmp.txt", "w+");
    FILE *dst = fopen("ans.txt", "w+");

    preprocess(src, tmp);
    

    return 0;
}

void addMacro(const char *name, const char *value) {
    if (macroCount >= 100) return;
    strcpy(macros[macroCount].name, name);
    strcpy(macros[macroCount].value, value);
    macroCount++;
}

const char* getMacroValue(const char *name) {
    for (int i = 0; i < macroCount; i++) {
        if (strcmp(macros[i].name, name) == 0)
            return macros[i].value;
    }
    return NULL;
}

void preprocess(FILE *src, FILE *dst) {
    int ch;
    char token[256];
    int tlen;

    while ((ch = fgetc(src)) != EOF) {

        if (ch == '/') {
            int next = fgetc(src);

            if (next == '/') {
                putc(' ', dst);
                putc(' ', dst);
                while ((ch = fgetc(src)) != EOF && ch != '\n')
                    putc(' ', dst);
                if (ch == '\n')
                    putc('\n', dst);
                continue;
            }

            if (next == '*') {
                putc(' ', dst);
                putc(' ', dst);
                int prev = 0;
                while ((ch = fgetc(src)) != EOF) {
                    if (ch == '\n')
                        putc('\n', dst);
                    else
                        putc(' ', dst);

                    if (prev == '*' && ch == '/')
                        break;
                    prev = ch;
                }
                continue;
            }

            putc('/', dst);
            ungetc(next, src);
            continue;
        }

        if (ch == '#') {
            char directive[20];
            int dlen = 0;

            directive[dlen++] = ch;

            while ((ch = fgetc(src)) != EOF && !isspace(ch)) {
                directive[dlen++] = ch;
            }
            directive[dlen] = '\0';

            if (strcmp(directive, "#include") == 0) {
                while(ch != '\n')
                    ch = fgetc(src);
                fputc(ch, dst);
                continue;
            } 
            if (strcmp(directive, "#define") == 0) {
                char name[100];
                char value[100];
                int i = 0;

                while (ch != EOF && isspace(ch))
                    ch = fgetc(src);

                while (ch != EOF && (isalnum(ch) || ch == '_')) {
                    name[i++] = ch;
                    ch = fgetc(src);
                }
                name[i] = '\0';

                while (ch != EOF && isspace(ch))
                    ch = fgetc(src);

                i = 0;
                while (ch != EOF && ch != '\n') {
                    value[i++] = ch;
                    ch = fgetc(src);
                }
                value[i] = '\0';

                addMacro(name, value);
                fputc(ch, dst);
                continue;
            }

            fputs(directive, dst);
            if (ch != EOF)
                putc(ch, dst);
            continue;
        }

        if (isalpha(ch) || ch == '_') {
            tlen = 0;
            token[tlen++] = ch;

            while ((ch = fgetc(src)) != EOF && (isalnum(ch) || ch == '_')) {
                token[tlen++] = ch;
            }
            token[tlen] = '\0';

            const char *val = getMacroValue(token);
            if (val)
                fputs(val, dst);
            else
                fputs(token, dst);

            if (ch != EOF)
                ungetc(ch, src);

            continue;
        }

        putc(ch, dst);
    }
}

token getNextToken(FILE *src, int *row, int *col){
    token curr;

    int ch;

    while (ch != EOF) {
        ch = fgetc(src);
        if (isalpha(ch)) {
            curr = isKeyword(ch, src, *row, *col);
            if(curr.token_name[0]) return curr;

            curr = isIdentifier(ch, src, *row, *col);
            if(curr.token_name[0]) return curr;
        }
        
        else if (isdigit(ch)) {
            curr = isNumber(ch, src, *row, *col);
            if(curr.token_name[0]) return curr;
        }
        
        else if (ch == '"') {
            curr = isStringLiteral(ch, src, *row, *col);
            if(curr.token_name[0]) return curr;
        }

        else if (strchr("+-&|*/%<>!^?", ch)) {
            curr = isOperator(ch, src, *row, *col);
            if(curr.token_name[0]) return curr;
        }

        else {
            curr = isSymbol(ch, src, *row, *col);
            if(curr.token_name[0]) return curr;
        }

    }
}