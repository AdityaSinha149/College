#ifndef LEXANALYSZER_H
#define LEXANALYSZER_H

#include "preprocessing.h"

// -------------------------
// Core token definitions
// -------------------------

typedef struct token {
    char tokenName[64];    // high-level category: keyword, id, number, op, symbol, etc.
    char lexeme[64];       // actual lexeme text (if you need it)
    int  row, col;         // position for error reporting
} token;

// Example keyword table (fill per grammar)
static const char *keywords[] = {
    // e.g.: "if", "else", "while", ...
};

// -------------------------
// Token classification API
// -------------------------

static token getNextToken(FILE *src, int *row, int *col);

static token isKeyword     (int ch, FILE *src, int *row, int *col);
static token isIdentifier  (int ch, FILE *src, int *row, int *col);
static token isNumber      (int ch, FILE *src, int *row, int *col);
static token isStringLit   (int ch, FILE *src, int *row, int *col);
static token isOperator    (int ch, FILE *src, int *row, int *col);
static token isSymbol      (int ch, FILE *src, int *row, int *col);

// Helpers
static void   printToken   (token t, FILE *dst);
static int    isKeywordLex (const char *word);

// -------------------------
// Token server (template)
// -------------------------

static token getNextToken(FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    int ch = fgetc(src);

    while (ch != EOF) {
        if (ch == ' ' || ch == '\t' || ch == '\r') {
            (*col)++;
            ch = fgetc(src);
            continue;
        }
        if (ch == '\n') {
            (*row)++;
            *col = 1;
            ch = fgetc(src);
            continue;
        }

        if (isalpha(ch) || ch == '_') {
            curr = isKeyword(ch, src, row, col);
            if (curr.tokenName[0]) return curr;

            curr = isIdentifier(ch, src, row, col);
            return curr;
        }

        if (isdigit(ch)) {
            curr = isNumber(ch, src, row, col);
            return curr;
        }

        if (ch == '"' || ch == '\'') {
            curr = isStringLit(ch, src, row, col);
            return curr;
        }

        if (strchr("+-&|*/%<>!^?=", ch)) {
            curr = isOperator(ch, src, row, col);
            return curr;
        }

        // Fallback: treat as single-character symbol/token
        curr = isSymbol(ch, src, row, col);
        return curr;
    }

    return curr; // EOF => empty token
}

// -------------------------
// Classification templates
// -------------------------

static token isKeyword(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    char word[64];
    int  len = 0;

    curr.row = *row;
    curr.col = *col;

    word[len++] = (char)ch;
    (*col)++;

    while ((ch = fgetc(src)) != EOF && (isalnum(ch) || ch == '_')) {
        if (len < (int)sizeof(word) - 1)
            word[len++] = (char)ch;
        (*col)++;
    }

    word[len] = '\0';

    if (ch != EOF)
        fseek(src, -1, SEEK_CUR);

    if (isKeywordLex(word)) {
        strcpy(curr.tokenName, "keyword");
        strcpy(curr.lexeme, word);
        return curr;
    }

    // Not a keyword: rewind and let isIdentifier() read again
    fseek(src, -(long)len, SEEK_CUR);
    *col -= len;
    return curr;
}

static token isIdentifier(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    char word[64];
    int  len = 0;

    curr.row = *row;
    curr.col = *col;

    word[len++] = (char)ch;
    (*col)++;

    while ((ch = fgetc(src)) != EOF && (isalnum(ch) || ch == '_')) {
        if (len < (int)sizeof(word) - 1)
            word[len++] = (char)ch;
        (*col)++;
    }

    word[len] = '\0';

    if (ch != EOF)
        fseek(src, -1, SEEK_CUR);

    strcpy(curr.tokenName, "id");
    strcpy(curr.lexeme, word);
    return curr;
}

static token isNumber(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    char buf[64];
    int  len = 0;

    curr.row = *row;
    curr.col = *col;

    buf[len++] = (char)ch;
    (*col)++;

    while ((ch = fgetc(src)) != EOF && isdigit(ch)) {
        if (len < (int)sizeof(buf) - 1)
            buf[len++] = (char)ch;
        (*col)++;
    }

    buf[len] = '\0';

    if (ch != EOF)
        fseek(src, -1, SEEK_CUR);

    strcpy(curr.tokenName, "number");
    strcpy(curr.lexeme, buf);
    return curr;
}

static token isStringLit(int quote, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    char buf[64];
    int  len = 0;

    curr.row = *row;
    curr.col = *col;

    int ch = fgetc(src);
    (*col)++;

    while (ch != EOF && ch != quote) {
        if (ch == '\n') {
            (*row)++;
            *col = 1;
        } else {
            if (len < (int)sizeof(buf) - 1)
                buf[len++] = (char)ch;
            (*col)++;
        }
        ch = fgetc(src);
    }

    buf[len] = '\0';
    if (ch != EOF) (*col)++;

    strcpy(curr.tokenName, "string");
    strcpy(curr.lexeme, buf);
    return curr;
}

static token isOperator(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    curr.row = *row;
    curr.col = *col;

    // TODO: extend with multi-character ops (>=, <=, &&, ||, etc.)
    curr.lexeme[0] = (char)ch;
    curr.lexeme[1] = '\0';

    strcpy(curr.tokenName, "op");
    (*col)++;
    return curr;
}

static token isSymbol(int ch, FILE *src, int *row, int *col) {
    token curr;
    memset(&curr, 0, sizeof(curr));

    curr.row = *row;
    curr.col = *col;

    curr.lexeme[0] = (char)ch;
    curr.lexeme[1] = '\0';
    strcpy(curr.tokenName, "symbol");
    (*col)++;
    return curr;
}

// -------------------------
// Helper implementations
// -------------------------

static int isKeywordLex(const char *word) {
    int n = sizeof(keywords) / sizeof(keywords[0]);
    for (int i = 0; i < n; i++) {
        if (strcmp(word, keywords[i]) == 0)
            return 1;
    }
    return 0;
}

static void printToken(token t, FILE *dst) {
    // Default token format: <TokenName,row,col>
    // You can change it to <lexeme,row,col> etc. per assignment.
    fprintf(dst, "<%s,%d,%d>", t.tokenName[0] ? t.tokenName : t.lexeme,
            t.row, t.col);
}

#endif
