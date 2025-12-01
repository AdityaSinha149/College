#include <stdio.h>
#include "mab.h"

int main() {
    MabPtr mem = createBlock(0, 100, 0);
    MabPtr a = memAlloc(mem, 30);
    printf("Allocated offset=%d size=%d\n", a->offset, a->size);
    MabPtr b = memAlloc(mem, 20);
    printf("Allocated offset=%d size=%d\n", b->offset, b->size);
    memFree(a);
    printf("Freed offset=%d\n", a->offset);
    MabPtr c = memAlloc(mem, 25);
    if (c) printf("Allocated offset=%d size=%d\n", c->offset, c->size);
    return 0;
}
