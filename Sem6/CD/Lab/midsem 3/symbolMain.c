#include "preprocessing.h"
#include "lexAnalyszer.h"
#include "symbolTable.h"

int main(void) {
    char filename[256];
    printf("Enter source file: ");
    if (scanf("%255s", filename) != 1) {
        fprintf(stderr, "No file name given.\n");
        return 1;
    }

    FILE *src = fopen(filename, "r");
    if (!src) {
        perror("open source");
        return 1;
    }

    FILE *tmp = fopen("tmp.txt", "w+");
    if (!tmp) {
        perror("tmp");
        fclose(src);
        return 1;
    }

    FILE *symOut = fopen("symtab.txt", "w+");
    if (!symOut) {
        perror("symtab");
        fclose(src);
        fclose(tmp);
        return 1;
    }

    // 1. Preprocess source into tmp.txt
    preprocess(src, tmp);
    fseek(tmp, 0, SEEK_SET);

    // 2. Initialise generic symbol table + scope stack
    SymbolTable st;
    ScopeStack  ss;
    stInit(&st);
    stScopeInit(&ss);

    // 3. Scan tokens and (later) build the symbol table.
    //    This loop is grammar-agnostic; you decide what to insert.
    int row = 1, col = 1;
    while (1) {
        token t = getNextToken(tmp, &row, &col);
        if (!t.tokenName[0] && !t.lexeme[0]) break; // EOF

        // TODO: Based on your grammar and token stream, decide when to:
        //   - stEnterScope(&ss);  // on '{' or similar
        //   - stExitScope(&ss);   // on '}' or similar
        //   - stInsert(&st, &ss, t.lexeme, "var"/"func"/..., "type");
        // For now, the body is intentionally empty so you can plug
        // in any grammar-specific logic you want.
    }

    // 4. Dump the symbol table
    stPrint(&st, symOut);

    fclose(src);
    fclose(tmp);
    fclose(symOut);
    return 0;
}
