// Simplified symbol table construction for the JS/jQuery midsem
// problem. We record function names (predefined jQuery functions
// used in the snippet) along with their argument list, and
// identifiers (including the special "#..." selectors) with
// their token type.

#include "lexAnalyszer.h"

#define MAX_SYMBOLS 100

typedef struct {
    char lexeme[50];     // Lexeme name
    char type[16];       // "FUNC" or "Identifier"
    char argument[256];  // Function argument string, empty for identifiers
} Symbol;

static Symbol table[MAX_SYMBOLS];
static int symbolCount = 0;

static const char *predefinedFuncs[] = {
    "ready", "on", "val", "split", "map",
    "filter", "text", "join", "isNaN", "isNan"
};

static int isPredefinedFunc(const char *name) {
    int n = sizeof(predefinedFuncs) / sizeof(predefinedFuncs[0]);
    for (int i = 0; i < n; i++) {
        if (strcmp(name, predefinedFuncs[i]) == 0)
            return 1;
    }
    return 0;
}

static int findSymbol(const char *lexeme, const char *type) {
    for (int i = 0; i < symbolCount; i++) {
        if (strcmp(table[i].lexeme, lexeme) == 0 &&
            strcmp(table[i].type, type) == 0) {
            return i;
        }
    }
    return -1;
}

static void trim(char *s) {
    int start = 0;
    while (s[start] && isspace((unsigned char)s[start])) start++;
    int end = (int)strlen(s) - 1;
    while (end >= start && isspace((unsigned char)s[end])) end--;
    int len = end - start + 1;
    if (len <= 0) {
        s[0] = '\0';
        return;
    }
    memmove(s, s + start, len);
    s[len] = '\0';
}

// Look ahead from the current file position to capture the
// argument list of a function call, without disturbing the
// main lexer stream. row/col are passed by value so that
// position tracking for tokens is unaffected.
static void extractArgument(FILE *src, int row, int col, char *out, size_t outSize) {
    long startPos = ftell(src);
    int ch;
    size_t idx = 0;

    // Skip whitespace
    while ((ch = fgetc(src)) != EOF) {
        if (ch == ' ' || ch == '\t' || ch == '\r' || ch == '\n') {
            if (ch == '\n') {
                row++;
                col = 1;
            } else {
                col++;
            }
            continue;
        }
        break;
    }

    if (ch != '(') {
        fseek(src, startPos, SEEK_SET);
        out[0] = '\0';
        return;
    }

    int depth = 1;
    while ((ch = fgetc(src)) != EOF && depth > 0) {
        if (ch == '\n') {
            row++;
            col = 1;
        } else {
            col++;
        }

        if (ch == '(') {
            depth++;
        } else if (ch == ')') {
            depth--;
            if (depth == 0) break;
        }

        if (depth > 0) {
            if (idx < outSize - 1) {
                out[idx++] = (char)ch;
            }
        }
    }

    out[idx] = '\0';
    trim(out);

    // Restore original position for the main scanning loop
    fseek(src, startPos, SEEK_SET);
}

static void addFunctionSymbol(FILE *src, int row, int col, const token *t) {
    if (symbolCount >= MAX_SYMBOLS) return;
    if (findSymbol(t->tokenValue, "FUNC") != -1) return; // avoid duplicates

    Symbol *s = &table[symbolCount++];
    strncpy(s->lexeme, t->tokenValue, sizeof(s->lexeme) - 1);
    s->lexeme[sizeof(s->lexeme) - 1] = '\0';
    strcpy(s->type, "FUNC");

    char arg[256];
    extractArgument(src, row, col, arg, sizeof(arg));
    strncpy(s->argument, arg, sizeof(s->argument) - 1);
    s->argument[sizeof(s->argument) - 1] = '\0';
}

static void addIdentifierSymbol(const token *t) {
    if (symbolCount >= MAX_SYMBOLS) return;
    if (findSymbol(t->tokenValue, "Identifier") != -1) return; // avoid duplicates

    Symbol *s = &table[symbolCount++];
    strncpy(s->lexeme, t->tokenValue, sizeof(s->lexeme) - 1);
    s->lexeme[sizeof(s->lexeme) - 1] = '\0';
    strcpy(s->type, "Identifier");
    s->argument[0] = '\0';
}

int main() {
    char file[100];
    scanf("%99s", file);

    FILE *src = fopen(file, "r");
    if (src == NULL) {
        perror("error");
        return 1;
    }

    FILE *tmp = fopen("tmp.txt", "w+");
    if (!tmp) {
        perror("error");
        fclose(src);
        return 1;
    }

    FILE *dst = fopen("ans.txt", "w+");
    if (!dst) {
        perror("error");
        fclose(src);
        fclose(tmp);
        return 1;
    }

    preprocess(src, tmp);
    fseek(tmp, 0, SEEK_SET);

    int row = 1, col = 1;
    while (1) {
        token t = getNextToken(tmp, &row, &col);
        if (!t.tokenName[0] && !t.tokenValue[0]) break;

        // Functions: predefined jQuery functions like ready, on, split, ...
        if (strcmp(t.tokenName, "id") == 0 && isPredefinedFunc(t.tokenValue)) {
            addFunctionSymbol(tmp, row, col, &t);
        }
        // Identifiers coming from selector strings ("#submit")
        else if (strcmp(t.tokenName, "Identifier") == 0) {
            addIdentifierSymbol(&t);
        }
        // Other identifiers such as input, numbers, divisibleByThree, num
        else if (strcmp(t.tokenName, "id") == 0) {
            addIdentifierSymbol(&t);
        }
    }

    // Print the symbol table as: Token Name, Token Type, Argument
    fprintf(dst, "%-15s %-12s %-s\n", "Token Name", "Token Type", "Argument");
    for (int i = 0; i < symbolCount; i++) {
        fprintf(dst, "%-15s %-12s %-s\n",
                table[i].lexeme,
                table[i].type,
                table[i].argument);
    }

    fclose(src);
    fclose(tmp);
    fclose(dst);
    return 0;
}
