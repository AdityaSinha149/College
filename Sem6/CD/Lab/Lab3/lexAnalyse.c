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
void copyFile(FILE *src, FILE *dst);
void lexAnalyse(FILE *src, FILE *dst);
int isKeyword(char *word);
int isDoubleOp(char ch, char next);
void printToken(token t, FILE *dst);

int main(){
    printf("Enter program to lexical analyse: ");
    char file[100];
    scanf("%s", file);

    FILE *src;
    src = fopen(file, "r");
    if(src == NULL){
        perror("error ");
        return 1;
    }
    FILE *tmp;
    tmp = fopen("tmp.txt", "w+");
    FILE *dst;
    dst = fopen("ans.txt", "w+");
    if(dst == NULL){
        perror("error ");
        fclose(src);
        return 1;
    }
    removePreprocessorDirectives(src, tmp);
    rewind(tmp);
    removeComments(tmp, dst);
    rewind(dst);
    copyFile(dst, tmp);
    rewind(tmp);
    removeExtraSpaces(tmp, dst);
    rewind(dst);
    copyFile(dst, tmp);
    rewind(tmp);
    lexAnalyse(tmp, dst);
}

void removeExtraSpaces(FILE *src, FILE *dst) {
    int ch;
    int state = 0;

    while ((ch = fgetc(src)) != EOF) {
        if (ch == ' ') {
            if (state == 0) fputc(ch, dst);
            state = 1;
        } else {
            state = 0;
            fputc(ch, dst);
        }
    }
}

void removePreprocessorDirectives(FILE *src, FILE *dst) {
    int ch;
    int state = 0;

    while ((ch = fgetc(src)) != EOF) {
        if (ch == '#') state = 1;
        if (ch == '\n') state = 0;
        if (state == 0) fputc(ch, dst);
    }
}

void removeComments(FILE *src, FILE *dst) {
    int ch, next;
    int state = 0;

    while ((ch = fgetc(src)) != EOF) {
        if (state == 0) {
            if (ch == '/') {
                next = fgetc(src);
                if (next == '/') {
                    state = 1;
                } else if (next == '*') {
                    state = 2;
                } else {
                    fputc(ch, dst);
                    if (next != EOF) fputc(next, dst);
                }
            } else {
                fputc(ch, dst);
            }
        } 
        else if (state == 1) {
            if (ch == '\n') {
                state = 0;
                fputc(ch, dst);
            }
        } 
        else if (state == 2) {
            if (ch == '*') {
                next = fgetc(src);
                if (next == '/') {
                    state = 0;
                } else {
                    if (next != EOF) ungetc(next, src);
                }
            }
        }
    }
}

void copyFile(FILE *src, FILE *dst) {
    int ch;
    while ((ch = fgetc(src)) != EOF) {
        fputc(ch, dst);
    }
}

void lexAnalyse(FILE *src, FILE *dst){
    /*
    state 0 : start
    state 1 : keyword & identifier
    state 2 : confirm identifier
    state 3 : nums
    state 4 : operators & symbols
    */

    int ch, i = 0;
    char word[1024];

    int state = 0;
    int row = 1, col = 0;

    token curr;

    while ((ch = fgetc(src)) != EOF) {
        col++;
        /*
        Token end cases:
        1. space
        2. alphabet after num or op or sym
        3. newline
        */
        if(ch == ' ' || (state > 2 && isalpha(ch)) ||
            ch == '\n' || (state == 5 && ch != '=')){
            
            printToken(curr, dst);
            
            //clear token
            while(i) word[i--] = 0;
            if(ch == '\n') {
                row++;
                col = 1;
                putc('\n', dst);
                continue;
            }
        }
        word[i++] = ch;

        curr.col = col;
        curr. row = row;

        if(state == 0){
            if(isalpha(ch)) state = 1;
            else if(isdigit(ch)) state = 3;
            else state = 4;
        }
        if(state == 1){
            if(isdigit(ch)) state = 2;

            if(isKeyword(word)) strcpy(curr.token_name, word);
            else strcpy(curr.token_name, "id");
        }
        if(state == 2){
            strcpy(curr.token_name, "id");
        }
        if(state == 3){
            strcpy(curr.token_name, "num");
        }
        if(state == 4){
            char next = fgetc(src);
            if(isDoubleOp(ch, next)){
                word[i++] = next;
                col++;
                curr.col++;
            }
            else ungetc(next, src);
            strcpy(curr.token_name, word);
        }
    }
}

int isKeyword(char *word)
{
    int count = sizeof(keywords) / sizeof(keywords[0]);

    for (int i = 0; i < count; i++) {
        if (strcmp(word, keywords[i]) == 0)
            return 1;
    }
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