#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

typedef struct token{
    char token_name [50];
    unsigned int row,col;
}token;

const char *keywords[] = {
    "auto", "break", "case", "char", "const",
    "continue", "default", "do", "double",
    "else", "enum", "extern", "float", "for",
    "goto", "if", "int", "long", "register",
    "return", "short", "signed", "sizeof",
    "static", "struct", "switch", "typedef",
    "union", "unsigned", "void", "volatile", "while"
};

void removeExtraSpaces(FILE *src, FILE *dst);
void removePreprocessorDirectives(FILE *src, FILE *dst);
void removeComments(FILE *src, FILE *dst);
void removeExtraLines(FILE *src, FILE *dst);
void copyFile(FILE *src, FILE *dst);
void lexAnalyse(FILE *src, FILE *dst);
int isKeyword(char *word);
int isDoubleOp(char ch, char next);
void printToken(token t, FILE *dst);

int main(){
    printf("Enter program to lexical analyse: ");
    char file[100];
    scanf("%s", file);

    FILE *src = fopen(file, "r");
    if(src == NULL){
        perror("error ");
        return 1;
    }

    FILE *tmp = fopen("tmp.txt", "w+");
    FILE *dst = fopen("ans.txt", "w+");

    removePreprocessorDirectives(src, tmp);
    fclose(tmp);

    tmp = fopen("tmp.txt", "r");
    removeComments(tmp, dst);
    fclose(tmp); fclose(dst);

    dst = fopen("ans.txt", "r");
    tmp = fopen("tmp.txt", "w");
    copyFile(dst, tmp);
    fclose(dst); fclose(tmp);

    tmp = fopen("tmp.txt", "r");
    dst = fopen("ans.txt", "w");
    removeExtraSpaces(tmp, dst);
    fclose(tmp); fclose(dst);

    dst = fopen("ans.txt", "r");
    tmp = fopen("tmp.txt", "w");
    copyFile(dst, tmp);
    fclose(dst); fclose(tmp);

    tmp = fopen("tmp.txt", "r");
    dst = fopen("ans.txt", "w");
    removeExtraLines(tmp, dst);
    fclose(tmp); fclose(dst);

    dst = fopen("ans.txt", "r");
    tmp = fopen("tmp.txt", "w");
    copyFile(dst, tmp);
    fclose(dst); fclose(tmp);

    tmp = fopen("tmp.txt", "r");
    dst = fopen("ans.txt", "w");
    lexAnalyse(tmp, dst);
    fclose(tmp); fclose(dst);

    return 0;
}

void removePreprocessorDirectives(FILE *src, FILE *dst) {
    int ch, state = 0;
    while ((ch = fgetc(src)) != EOF) {
        if (ch == '#') state = 1;
        if (ch == '\n') state = 0;
        if (state == 0) fputc(ch, dst);
    }
}

void removeComments(FILE *src, FILE *dst) {
    int ch, next, state = 0;
    while ((ch = fgetc(src)) != EOF) {
        if (state == 0) {
            if (ch == '/') {
                next = fgetc(src);
                if (next == '/') state = 1;
                else if (next == '*') state = 2;
                else {
                    fputc(ch, dst);
                    if (next != EOF) fputc(next, dst);
                }
            } else fputc(ch, dst);
        }
        else if (state == 1) {
            if (ch == '\n') { state = 0; fputc(ch, dst); }
        }
        else if (state == 2) {
            if (ch == '*') {
                next = fgetc(src);
                if (next == '/') state = 0;
                else if (next != EOF) ungetc(next, src);
            }
        }
    }
}

void removeExtraSpaces(FILE *src, FILE *dst) {
    int ch, startOfLine = 1;
    while ((ch = fgetc(src)) != EOF) {
        if(ch == '"'){
            fputc(ch, dst);
            while((ch = fgetc(src)) != '"') fputc(ch, dst);
        }
        if (ch == ' ') {
            while ((ch = fgetc(src)) == ' ');
            if (!startOfLine) fputc(' ', dst);
            if (ch != EOF) fputc(ch, dst);
            startOfLine = 0;
        } else {
            fputc(ch, dst);
            startOfLine = (ch == '\n');
        }
    }
}

void removeExtraLines(FILE *src, FILE *dst){
    int ch, newline = 1;
    while((ch = fgetc(src)) != EOF){
        if(ch == '\n'){
            if(newline == 1){
                while((ch = fgetc(src)) == '\n');
                if (ch != EOF) fputc(ch, dst);
            }
            else{
                newline = 1;
                fputc(ch, dst);
            }
        }
        else{
            newline = 0;
            fputc(ch, dst);
        }
    }
}

void copyFile(FILE *src, FILE *dst) {
    int ch;
    while ((ch = fgetc(src)) != EOF) fputc(ch, dst);
}

void lexAnalyse(FILE *src, FILE *dst){

    int ch, i = 0;
    char word[1024];
    int state = 0, stateChange = 0;
    token curr;
    int row = 1, col = 0;

    memset(&curr, 0, sizeof(curr));
    memset(word, 0, sizeof(word));

    /*
    state 0 : start
    state 1 : keyword & identifier
    state 2 : confirm identifier
    state 3 : nums
    state 4 : string literals
    state 5 : operators & symbols
    */

    ch = fgetc(src);
    while (ch != EOF) {

        col++;

        if (state == 0) {
            curr.row = row;
            curr.col = col;

            if (isalpha(ch)) {
                word[i++] = ch;
                state = 1;
            }
            else if (isdigit(ch)) {
                word[i++] = ch;
                state = 3;
            }
            else if (ch == '"') {
                word[i++] = ch;
                state = 4;
            }
            else if (ch != ' ' && ch != '\n' && ch != '\t') {
                word[i++] = ch;
                state = 5;
            }

            if (state != 0)
                stateChange = 1;
        }

        else if (state == 1) {
            if (isalpha(ch)) word[i++] = ch;
            else if (isdigit(ch)) { word[i++] = ch; state = 2; }
            else {
                word[i] = '\0';
                strcpy(curr.token_name, isKeyword(word) ? word : "id");
                printToken(curr, dst);
                memset(word, 0, sizeof(word));
                i = 0;
                ungetc(ch, src);
                col--;
                state = 0;
            }
        }

        else if (state == 2) {
            if (isalnum(ch)) word[i++] = ch;
            else {
                word[i] = '\0';
                strcpy(curr.token_name, "id");
                printToken(curr, dst);
                memset(word, 0, sizeof(word));
                i = 0;
                ungetc(ch, src);
                state = 0;
            }
        }

        else if (state == 3) {
            if (isdigit(ch)) word[i++] = ch;
            else {
                word[i] = '\0';
                strcpy(curr.token_name, "num");
                printToken(curr, dst);
                memset(word, 0, sizeof(word));
                i = 0;
                ungetc(ch, src);
                state = 0;
            }
        }

        else if (state == 4) {
            word[i++] = ch;
            if (ch == '"') {
                word[i] = '\0';
                strcpy(curr.token_name, "string");
                printToken(curr, dst);
                memset(word, 0, sizeof(word));
                i = 0;
                state = 0;
            }
        }

        else if (state == 5) {
            char next = fgetc(src);

            if (isDoubleOp(word[0], next)) {
                word[i++] = next;
                col++;
            }
            else {
                fseek(src, -1, SEEK_CUR);
            }

            word[i] = '\0';
            strcpy(curr.token_name, word);
            printToken(curr, dst);
            memset(word, 0, sizeof(word));
            i = 0;
            state = 0;
            stateChange = 0;

        }

        if (stateChange == 0)
            ch = fgetc(src);
        else
            stateChange = 0;

        if (ch == '\n') {
            row++;
            col = 0;
            fputc(ch, dst);
        }
    }

    if (i > 0 && state != 0) {
        word[i] = '\0';
        if (state == 1) strcpy(curr.token_name, isKeyword(word) ? word : "id");
        else if (state == 2) strcpy(curr.token_name, "id");
        else if (state == 3) strcpy(curr.token_name, "num");
        else if (state == 4) strcpy(curr.token_name, "string");
        else strcpy(curr.token_name, word);
        printToken(curr, dst);
    }
}

int isKeyword(char *word) {
    int count = sizeof(keywords) / sizeof(keywords[0]);
    for (int i = 0; i < count; i++)
        if (strcmp(word, keywords[i]) == 0) return 1;
    return 0;
}

int isDoubleOp(char ch, char next) {
    if (ch == '=' && next == '=') return 1;
    if (ch == '!' && next == '=') return 1;
    if (ch == '<' && next == '=') return 1;
    if (ch == '>' && next == '=') return 1;
    if (ch == '&' && next == '&') return 1;
    if (ch == '|' && next == '|') return 1;
    if (ch == '+' && next == '+') return 1;
    if (ch == '-' && next == '-') return 1;
    if (ch == '+' && next == '=') return 1;
    if (ch == '-' && next == '=') return 1;
    if (ch == '*' && next == '=') return 1;
    if (ch == '/' && next == '=') return 1;
    if (ch == '%' && next == '=') return 1;
    return 0;
}

void printToken(token t, FILE *dst){
    fprintf(dst,"<%s,%d,%d>", t.token_name, t.row, t.col);
}
