#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

// For this midsem JS/jQuery problem the input file already
// contains just the source code we want to analyse, so the
// preprocessor only needs to make a straight copy. Newlines
// are preserved so that row/column tracking in the lexer
// remains correct.
static void preprocess(FILE *src, FILE *dst) {
    int ch;
    while ((ch = fgetc(src)) != EOF) {
        fputc(ch, dst);
    }
}

#endif
