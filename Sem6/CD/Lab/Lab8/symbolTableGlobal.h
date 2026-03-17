#ifndef SYMBOL_TABLE_GLOBAL_H
#define SYMBOL_TABLE_GLOBAL_H

#include "lexAnalyszer.h"

#define TABLE_SIZE 101

typedef struct symbolTable symbolTable;

typedef struct symbolTableEntry {
    token tokenInfo;
    struct symbolTableEntry *nextTokenEntry;
    symbolTable *local;
} symbolTableEntry;

struct symbolTable {
    symbolTableEntry *entry[TABLE_SIZE];
};

static void initSymbolTable(symbolTable *st);
static symbolTableEntry *findSymbol(symbolTable *st, const char *name);
static symbolTableEntry *addVariableSymbol(symbolTable *st, const char *name, const char *typeName, int size);
static symbolTableEntry *addFunctionSymbol(symbolTable *st, const char *name, const char *returnType);
static void printLocalSymbolTable(symbolTable *st, FILE *dst);
static void printGlobalSymbolTable(symbolTable *st, FILE *dst);
static void freeSymbolTable(symbolTable *st);

static unsigned int symbolHash(const char *name) {
    unsigned int hashValue = 0;

    while (*name != '\0') {
        hashValue = (hashValue * 131u) + (unsigned char)(*name);
        name++;
    }

    return hashValue % TABLE_SIZE;
}

static void initSymbolTable(symbolTable *st) {
    memset(st, 0, sizeof(*st));
}

static symbolTableEntry *createSymbolEntry(const token *info) {
    symbolTableEntry *entry = (symbolTableEntry *)calloc(1, sizeof(symbolTableEntry));

    if (entry == NULL)
        return NULL;

    entry->tokenInfo = *info;
    return entry;
}

static symbolTableEntry *findSymbol(symbolTable *st, const char *name) {
    unsigned int index;
    symbolTableEntry *entry;

    if (st == NULL || name == NULL)
        return NULL;

    index = symbolHash(name);
    entry = st->entry[index];

    while (entry != NULL) {
        if (strcmp(entry->tokenInfo.tokenValue, name) == 0)
            return entry;
        entry = entry->nextTokenEntry;
    }

    return NULL;
}

static symbolTableEntry *insertSymbol(symbolTable *st, const token *info) {
    unsigned int index;
    symbolTableEntry *entry;

    if (st == NULL || info == NULL)
        return NULL;

    entry = findSymbol(st, info->tokenValue);
    if (entry != NULL)
        return entry;

    entry = createSymbolEntry(info);
    if (entry == NULL)
        return NULL;

    index = symbolHash(info->tokenValue);
    entry->nextTokenEntry = st->entry[index];
    st->entry[index] = entry;
    return entry;
}

static symbolTableEntry *addVariableSymbol(symbolTable *st, const char *name, const char *typeName, int size) {
    token info;

    memset(&info, 0, sizeof(info));
    strcpy(info.tokenName, "id");
    strcpy(info.tokenValue, name);
    strcpy(info.tokenType, typeName);
    info.size = size;

    return insertSymbol(st, &info);
}

static symbolTableEntry *addFunctionSymbol(symbolTable *st, const char *name, const char *returnType) {
    symbolTableEntry *entry;
    token info;

    memset(&info, 0, sizeof(info));
    strcpy(info.tokenName, "Func");
    strcpy(info.tokenValue, name);
    strcpy(info.tokenReturnType, returnType);

    entry = insertSymbol(st, &info);
    if (entry == NULL)
        return NULL;

    if (entry->local == NULL) {
        entry->local = (symbolTable *)calloc(1, sizeof(symbolTable));
        if (entry->local == NULL)
            return NULL;
    }

    return entry;
}

static void printLocalSymbolTable(symbolTable *st, FILE *dst) {
    int idx = 1;

    fprintf(dst, "%-6s %-12s %-12s %-6s %-12s\n",
            "", "Name", "Type", "Size", "Return Type");

    for (int i = 0; i < TABLE_SIZE; i++) {
        symbolTableEntry *entry = st->entry[i];

        while (entry != NULL) {
            char sizeStr[20];

            if (entry->tokenInfo.size > 0)
                snprintf(sizeStr, sizeof(sizeStr), "%d", entry->tokenInfo.size);
            else
                strcpy(sizeStr, "-");

            fprintf(dst, "%-6d %-12s %-12s %-6s %-12s\n",
                    idx++,
                    entry->tokenInfo.tokenValue[0] ? entry->tokenInfo.tokenValue : "-",
                    entry->tokenInfo.tokenType[0] ? entry->tokenInfo.tokenType : entry->tokenInfo.tokenName,
                    sizeStr,
                    entry->tokenInfo.tokenReturnType[0] ? entry->tokenInfo.tokenReturnType : "-");

            entry = entry->nextTokenEntry;
        }
    }
}

static void printGlobalSymbolTable(symbolTable *st, FILE *dst) {
    int idx = 1;

    fprintf(dst, "%-6s %-12s %-12s %-6s %-12s %-18s\n",
            "", "Name", "Type", "Size", "Return Type", "Ptr to Local Table");

    for (int i = 0; i < TABLE_SIZE; i++) {
        symbolTableEntry *entry = st->entry[i];

        while (entry != NULL) {
            char sizeStr[20];
            char localPtrStr[20];

            if (entry->tokenInfo.size > 0)
                snprintf(sizeStr, sizeof(sizeStr), "%d", entry->tokenInfo.size);
            else
                strcpy(sizeStr, "-");

            if (entry->local != NULL)
                snprintf(localPtrStr, sizeof(localPtrStr), "%p", (void *)entry->local);
            else
                strcpy(localPtrStr, "-");

            fprintf(dst, "%-6d %-12s %-12s %-6s %-12s %-18s\n",
                    idx++,
                    entry->tokenInfo.tokenValue[0] ? entry->tokenInfo.tokenValue : "-",
                    entry->tokenInfo.tokenName[0] ? entry->tokenInfo.tokenName : "-",
                    sizeStr,
                    entry->tokenInfo.tokenReturnType[0] ? entry->tokenInfo.tokenReturnType : "-",
                    localPtrStr);

            entry = entry->nextTokenEntry;
        }
    }

    fprintf(dst, "\n");

    for (int i = 0; i < TABLE_SIZE; i++) {
        symbolTableEntry *entry = st->entry[i];

        while (entry != NULL) {
            if (entry->local != NULL) {
                fprintf(dst, "%s's Local Table:\n", entry->tokenInfo.tokenValue);
                printLocalSymbolTable(entry->local, dst);
                fprintf(dst, "\n");
            }
            entry = entry->nextTokenEntry;
        }
    }
}

static void freeSymbolTable(symbolTable *st) {
    for (int i = 0; i < TABLE_SIZE; i++) {
        symbolTableEntry *entry = st->entry[i];

        while (entry != NULL) {
            symbolTableEntry *next = entry->nextTokenEntry;

            if (entry->local != NULL) {
                freeSymbolTable(entry->local);
                free(entry->local);
            }

            free(entry);
            entry = next;
        }

        st->entry[i] = NULL;
    }
}

#endif
