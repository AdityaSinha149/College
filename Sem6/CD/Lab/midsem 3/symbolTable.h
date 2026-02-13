#ifndef SYMBOL_TABLE_H
#define SYMBOL_TABLE_H

// Use the same lexer header name as in this folder
// (see lexAnalyszer.h). This gives access to the token
// definition if you want to store extra info from tokens
// into the symbol table in future.
#include "lexAnalyszer.h"

// -------------------------
// Generic symbol-table template
// -------------------------
// Supports:
//  - Global and (optionally) local scopes via a stack
//  - Insertion / lookup of identifiers or non-terminals
//  - You can extend entries with type, attributes, etc.

#define ST_TABLE_SIZE 101

typedef struct SymbolEntry {
    char  name[64];
    char  kind[32];     // e.g., "var", "func", "temp", "nonterm", ...
    char  type[32];     // semantic type or grammar category
    int   scopeLevel;   // for nested scopes
    struct SymbolEntry *next;
} SymbolEntry;

typedef struct SymbolTable {
    SymbolEntry *buckets[ST_TABLE_SIZE];
} SymbolTable;

// Scope stack (very simple; extend if you need more data per scope)

typedef struct ScopeStack {
    int  level;         // current nesting level
} ScopeStack;

// API
static void stInit       (SymbolTable *st);
static void stScopeInit  (ScopeStack *ss);
static void stEnterScope (ScopeStack *ss);
static void stExitScope  (ScopeStack *ss);

static SymbolEntry *stInsert(SymbolTable *st, ScopeStack *ss,
                             const char *name,
                             const char *kind,
                             const char *type);

static SymbolEntry *stLookup(SymbolTable *st, const char *name);
static int          stHash  (const char *name);
static void         stPrint (SymbolTable *st, FILE *dst);

// -------------------------
// Implementation templates
// -------------------------

static void stInit(SymbolTable *st) {
    memset(st, 0, sizeof(*st));
}

static void stScopeInit(ScopeStack *ss) {
    ss->level = 0;
}

static void stEnterScope(ScopeStack *ss) {
    ss->level++;
}

static void stExitScope(ScopeStack *ss) {
    if (ss->level > 0) ss->level--;
}

static int stHash(const char *name) {
    unsigned h = 0;
    for (const unsigned char *p = (const unsigned char *)name; *p; ++p)
        h = h * 31u + *p;
    return (int)(h % ST_TABLE_SIZE);
}

static SymbolEntry *stInsert(SymbolTable *st, ScopeStack *ss,
                             const char *name,
                             const char *kind,
                             const char *type) {
    int idx = stHash(name);

    SymbolEntry *e = (SymbolEntry *)malloc(sizeof(SymbolEntry));
    memset(e, 0, sizeof(SymbolEntry));
    strncpy(e->name, name, sizeof(e->name) - 1);
    strncpy(e->kind, kind, sizeof(e->kind) - 1);
    strncpy(e->type, type, sizeof(e->type) - 1);
    e->scopeLevel = ss ? ss->level : 0;

    e->next = st->buckets[idx];
    st->buckets[idx] = e;
    return e;
}

static SymbolEntry *stLookup(SymbolTable *st, const char *name) {
    int idx = stHash(name);
    SymbolEntry *e = st->buckets[idx];
    while (e) {
        if (strcmp(e->name, name) == 0)
            return e;
        e = e->next;
    }
    return NULL;
}

static void stPrint(SymbolTable *st, FILE *dst) {
    fprintf(dst, "%-4s %-16s %-10s %-10s\n", "S", "Name", "Kind", "Type");
    int serial = 1;
    for (int i = 0; i < ST_TABLE_SIZE; i++) {
        SymbolEntry *e = st->buckets[i];
        while (e) {
            fprintf(dst, "%-4d %-16s %-10s %-10s\n",
                    serial++, e->name, e->kind, e->type);
            e = e->next;
        }
    }
}

#endif
