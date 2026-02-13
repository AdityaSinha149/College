#ifndef PREPROCESSING_H
#define PREPROCESSING_H

#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

// Generic preprocessing stage.
// Typical responsibilities (you can extend/change this per grammar):
//  - Remove / normalise comments
//  - Handle macro-like directives
//  - Normalise whitespace while preserving line numbers
//  - Possibly expand includes or custom directives
//
// For now this is just a template: fill in the body of preprocess()
// for your particular source language / grammar.

typedef struct {
    char name[100];
    char value[100];
} Macro;

static Macro macros[100];
static int macroCount = 0;

static void addMacro(const char *name, const char *value);
static const char *getMacroValue(const char *name);
static void preprocess(FILE *src, FILE *dst);

// --- implementation templates ---

static void preprocess(FILE *src, FILE *dst) {
    int ch;
    // TODO: implement comment removal / macro handling etc.
    // For now: simple copy that preserves newlines.
    while ((ch = fgetc(src)) != EOF) {
        fputc(ch, dst);
    }
}

static void addMacro(const char *name, const char *value) {
    // TODO: implement a real macro table if your grammar needs it.
    if (macroCount >= 100) return;
    strcpy(macros[macroCount].name, name);
    strcpy(macros[macroCount].value, value);
    macroCount++;
}

static const char *getMacroValue(const char *name) {
    // TODO: implement real lookup rules (e.g., #define style).
    for (int i = 0; i < macroCount; i++) {
        if (strcmp(macros[i].name, name) == 0)
            return macros[i].value;
    }
    return NULL;
}

#endif
