#include "preprocessing.h"
#include "lexer.h"

#include <stdio.h>
#include <stdlib.h>

static const char *tokenTypeName(TokenType type) {
    switch (type) {
        case TOKEN_EOF: return "EOF";
        case TOKEN_KEYWORD: return "KEYWORD";
        case TOKEN_IDENTIFIER: return "IDENTIFIER";
        case TOKEN_NUMBER: return "NUMBER";
        case TOKEN_STRING: return "STRING";
        case TOKEN_OPERATOR: return "OPERATOR";
        case TOKEN_DELIMITER: return "DELIMITER";
        default: return "UNKNOWN";
    }
}

int main(int argc, char **argv) {
    if (argc < 2) {
        printf("Usage: %s <source-file>\n", argv[0]);
        return 1;
    }

    char *source = preprocessSourceFile(argv[1]);
    if (!source) {
        printf("Failed to read file: %s\n", argv[1]);
        return 1;
    }

    initLexer(source);

    printf("Tokens:\n");
    while (1) {
        Token t = getNextToken();
        printf("%-10s | %-20s | line %d col %d\n", tokenTypeName(t.type), t.lexeme, t.line, t.column);
        if (t.type == TOKEN_EOF) {
            break;
        }
    }

    free(source);
    return 0;
}
