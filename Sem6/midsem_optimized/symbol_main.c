#include "preprocessing.h"
#include "lexer.h"
#include "symbolTable.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int isPotentialFunctionName(const Token *t) {
    if (!t) {
        return 0;
    }
    return t->type == TOKEN_IDENTIFIER || t->type == TOKEN_KEYWORD;
}

static void collectArguments(char *outArgs, size_t outSize) {
    size_t used = 0;
    int depth = 1;

    while (depth > 0) {
        Token t = getNextToken();
        if (t.type == TOKEN_EOF) {
            break;
        }
        if (t.type == TOKEN_DELIMITER && strcmp(t.lexeme, "(") == 0) {
            depth++;
        } else if (t.type == TOKEN_DELIMITER && strcmp(t.lexeme, ")") == 0) {
            depth--;
            if (depth == 0) {
                break;
            }
        }

        if (depth > 0) {
            size_t len = strlen(t.lexeme);
            if (used + len + 2 < outSize) {
                if (used > 0) {
                    outArgs[used++] = ' ';
                }
                memcpy(&outArgs[used], t.lexeme, len);
                used += len;
                outArgs[used] = '\0';
            }
        }
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

    SymbolTable table;
    initSymbolTable(&table);

    initLexer(source);

    while (1) {
        Token t = getNextToken();
        if (t.type == TOKEN_EOF) {
            break;
        }

        if (isPotentialFunctionName(&t)) {
            Token next = getNextToken();
            if (next.type == TOKEN_DELIMITER && strcmp(next.lexeme, "(") == 0) {
                char args[MAX_ARGS_LEN] = {0};
                collectArguments(args, sizeof(args));
                addFunctionSymbol(&table, t.lexeme, args);
            }
        }
    }

    printSymbolTable(&table);
    free(source);
    return 0;
}
