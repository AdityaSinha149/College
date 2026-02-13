#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

#define MAX_SYMBOLS 256
#define MAX_NAME_LEN 128
#define MAX_ARGS_LEN 256

typedef struct SymbolEntry {
    char name[MAX_NAME_LEN];
    char args[MAX_ARGS_LEN];
} SymbolEntry;

typedef struct SymbolTable {
    SymbolEntry entries[MAX_SYMBOLS];
    size_t count;
} SymbolTable;

static void initSymbolTable(SymbolTable *table);
static int addFunctionSymbol(SymbolTable *table, const char *name, const char *args);
static void printSymbolTable(const SymbolTable *table);

static void initSymbolTable(SymbolTable *table) {
    if (!table) {
        return;
    }
    table->count = 0;
}

static int addFunctionSymbol(SymbolTable *table, const char *name, const char *args) {
    if (!table || !name || !args) {
        return 0;
    }
    if (table->count >= MAX_SYMBOLS) {
        return 0;
    }
    for (size_t i = 0; i < table->count; i++) {
        if (strcmp(table->entries[i].name, name) == 0 &&
            strcmp(table->entries[i].args, args) == 0) {
            return 1;
        }
    }
    strncpy(table->entries[table->count].name, name, MAX_NAME_LEN - 1);
    table->entries[table->count].name[MAX_NAME_LEN - 1] = '\0';
    strncpy(table->entries[table->count].args, args, MAX_ARGS_LEN - 1);
    table->entries[table->count].args[MAX_ARGS_LEN - 1] = '\0';
    table->count++;
    return 1;
}

static void printSymbolTable(const SymbolTable *table) {
    if (!table) {
        return;
    }
    printf("\nSymbol Table (Function name | Arguments)\n");
    printf("--------------------------------------\n");
    for (size_t i = 0; i < table->count; i++) {
        printf("%-20s | %s\n", table->entries[i].name, table->entries[i].args);
    }
}

#endif
