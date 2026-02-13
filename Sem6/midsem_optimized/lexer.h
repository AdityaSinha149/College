#ifndef LEXER_H
#define LEXER_H

#include <stddef.h>
#include <ctype.h>
#include <string.h>

typedef enum TokenType {
    TOKEN_EOF = 0,
    TOKEN_KEYWORD,
    TOKEN_IDENTIFIER,
    TOKEN_NUMBER,
    TOKEN_STRING,
    TOKEN_OPERATOR,
    TOKEN_DELIMITER
} TokenType;

typedef struct Token {
    TokenType type;
    char lexeme[256];
    int line;
    int column;
} Token;

// Initializes lexer with source buffer. The buffer must remain valid.
static void initLexer(const char *source);

// Returns the next token from the stream.
static Token getNextToken(void);

// Helper predicates
static int isKeyword(const char *lexeme);
static int isIdentifierStart(char c);
static int isIdentifierPart(char c);
static int isNumberStart(char c);
static int isOperatorChar(char c);
static int isDelimiterChar(char c);
static int isWhitespace(char c);

static const char *g_source = NULL;
static size_t g_pos = 0;
static int g_line = 1;
static int g_col = 1;

static void initLexer(const char *source) {
    g_source = source ? source : "";
    g_pos = 0;
    g_line = 1;
    g_col = 1;
}

static char peekChar(void) {
    return g_source[g_pos];
}

static char getChar(void) {
    char c = g_source[g_pos];
    if (c == '\0') {
        return c;
    }
    g_pos++;
    if (c == '\n') {
        g_line++;
        g_col = 1;
    } else {
        g_col++;
    }
    return c;
}

static void skipWhitespace(void) {
    while (isWhitespace(peekChar())) {
        getChar();
    }
}

static int isWhitespace(char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int isKeyword(const char *lexeme) {
    static const char *keywords[] = {
        "if", "else", "while", "for", "foreach", "function", "return",
        "echo", "assert", "true", "false", "null"
    };
    size_t count = sizeof(keywords) / sizeof(keywords[0]);
    for (size_t i = 0; i < count; i++) {
        if (strcmp(lexeme, keywords[i]) == 0) {
            return 1;
        }
    }
    return 0;
}

static int isIdentifierStart(char c) {
    return isalpha((unsigned char)c) || c == '_' || c == '$';
}

static int isIdentifierPart(char c) {
    return isalnum((unsigned char)c) || c == '_' || c == '$';
}

static int isNumberStart(char c) {
    return isdigit((unsigned char)c);
}

static int isOperatorChar(char c) {
    return strchr("+-*/=%!<>&|", c) != NULL;
}

static int isDelimiterChar(char c) {
    return strchr(";:,(){}[]", c) != NULL;
}

static Token makeToken(TokenType type, const char *lexeme, int line, int column) {
    Token t;
    t.type = type;
    t.line = line;
    t.column = column;
    strncpy(t.lexeme, lexeme, sizeof(t.lexeme) - 1);
    t.lexeme[sizeof(t.lexeme) - 1] = '\0';
    return t;
}

static Token readNumber(void) {
    char buf[256];
    size_t i = 0;
    int line = g_line;
    int col = g_col;

    while (isNumberStart(peekChar()) && i < sizeof(buf) - 1) {
        buf[i++] = getChar();
    }
    if (peekChar() == '.' && i < sizeof(buf) - 1) {
        buf[i++] = getChar();
        while (isNumberStart(peekChar()) && i < sizeof(buf) - 1) {
            buf[i++] = getChar();
        }
    }
    buf[i] = '\0';
    return makeToken(TOKEN_NUMBER, buf, line, col);
}

static Token readIdentifierOrKeyword(void) {
    char buf[256];
    size_t i = 0;
    int line = g_line;
    int col = g_col;

    while (isIdentifierPart(peekChar()) && i < sizeof(buf) - 1) {
        buf[i++] = getChar();
    }
    buf[i] = '\0';

    if (isKeyword(buf)) {
        return makeToken(TOKEN_KEYWORD, buf, line, col);
    }
    return makeToken(TOKEN_IDENTIFIER, buf, line, col);
}

static Token readString(void) {
    char buf[256];
    size_t i = 0;
    int line = g_line;
    int col = g_col;

    char quote = getChar();
    (void)quote;

    while (peekChar() != '\0' && peekChar() != '"' && i < sizeof(buf) - 1) {
        char c = getChar();
        if (c == '\\' && peekChar() != '\0' && i < sizeof(buf) - 1) {
            buf[i++] = c;
            c = getChar();
        }
        buf[i++] = c;
    }

    if (peekChar() == '"') {
        getChar();
    }

    buf[i] = '\0';
    return makeToken(TOKEN_STRING, buf, line, col);
}

static Token readOperator(void) {
    char buf[4];
    size_t i = 0;
    int line = g_line;
    int col = g_col;

    char c = getChar();
    buf[i++] = c;
    if ((c == '=' || c == '!' || c == '<' || c == '>') && peekChar() == '=') {
        buf[i++] = getChar();
    } else if ((c == '&' && peekChar() == '&') || (c == '|' && peekChar() == '|')) {
        buf[i++] = getChar();
    }
    buf[i] = '\0';
    return makeToken(TOKEN_OPERATOR, buf, line, col);
}

static Token getNextToken(void) {
    skipWhitespace();

    int line = g_line;
    int col = g_col;
    char c = peekChar();

    if (c == '\0') {
        return makeToken(TOKEN_EOF, "EOF", line, col);
    }

    if (isNumberStart(c)) {
        return readNumber();
    }

    if (isIdentifierStart(c)) {
        return readIdentifierOrKeyword();
    }

    if (c == '"') {
        return readString();
    }

    if (isOperatorChar(c)) {
        return readOperator();
    }

    if (isDelimiterChar(c)) {
        char buf[2];
        buf[0] = getChar();
        buf[1] = '\0';
        return makeToken(TOKEN_DELIMITER, buf, line, col);
    }

    // Unknown char: skip it and return as delimiter
    char buf[2];
    buf[0] = getChar();
    buf[1] = '\0';
    return makeToken(TOKEN_DELIMITER, buf, line, col);
}

#endif
