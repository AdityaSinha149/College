#include "preprocessing.h"
#include "lexAnalyszer.h"

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

    FILE *tokOut = fopen("tokens.txt", "w+");
    if (!tokOut) {
        perror("tokens");
        fclose(src);
        fclose(tmp);
        return 1;
    }

    // 1. Preprocess source into tmp.txt
    preprocess(src, tmp);
    fseek(tmp, 0, SEEK_SET);

    // 2. Lexical analysis: dump tokens
    int row = 1, col = 1;
    while (1) {
        token t = getNextToken(tmp, &row, &col);
        if (!t.tokenName[0] && !t.lexeme[0]) break; // EOF

        // You can change printToken to use lexeme instead of tokenName
        // depending on the assignment.
        printToken(t, tokOut);
    }

    fclose(src);
    fclose(tmp);
    fclose(tokOut);
    return 0;
}
