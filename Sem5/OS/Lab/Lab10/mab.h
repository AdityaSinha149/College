#ifndef MAB_H
#define MAB_H

#include <stdio.h>
#include <stdlib.h>

typedef struct mab {
    int offset;
    int size;
    int allocated;
    struct mab *next;
    struct mab *prev;
} Mab, *MabPtr;

static MabPtr createBlock(int offset, int size, int allocated) {
    MabPtr block = (MabPtr) malloc(sizeof(Mab));
    if (!block) exit(1);
    block->offset = offset;
    block->size = size;
    block->allocated = allocated;
    block->next = NULL;
    block->prev = NULL;
    return block;
}

static MabPtr memChk(MabPtr m, int size) {
    MabPtr curr = m;
    while (curr) {
        if (!curr->allocated && curr->size >= size) return curr;
        curr = curr->next;
    }
    return NULL;
}

static MabPtr memSplit(MabPtr m, int size) {
    if (m->size <= size) return m;
    MabPtr newBlock = createBlock(m->offset + size, m->size - size, 0);
    newBlock->next = m->next;
    if (newBlock->next) newBlock->next->prev = newBlock;
    newBlock->prev = m;
    m->next = newBlock;
    m->size = size;
    return m;
}

static MabPtr memMerge(MabPtr m) {
    if (!m) return NULL;
    while (m->next && !m->allocated && !m->next->allocated) {
        MabPtr nxt = m->next;
        m->size += nxt->size;
        m->next = nxt->next;
        if (m->next) m->next->prev = m;
        free(nxt);
    }
    while (m->prev && !m->allocated && !m->prev->allocated) {
        MabPtr prev = m->prev;
        prev->size += m->size;
        prev->next = m->next;
        if (m->next) m->next->prev = prev;
        free(m);
        m = prev;
    }
    return m;
}

static MabPtr memFree(MabPtr m) {
    if (!m) return NULL;
    m->allocated = 0;
    return memMerge(m);
}

static MabPtr memAlloc(MabPtr m, int size) {
    MabPtr curr = m;
    while (curr) {
        if (!curr->allocated && curr->size >= size) {
            curr = memSplit(curr, size);
            curr->allocated = 1;
            return curr;
        }
        curr = curr->next;
    }
    return NULL;
}

#endif
