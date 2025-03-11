#include <stdio.h>
#include <stdlib.h>

int n;

int hash(int key) {
    return key % n;
}

int probe(int H[], int key) {
    int index = hash(key);
    int i = 0;
    while (H[(index + i) % n] != 0)
        i++;
    return (index + i) % n;
}

void insert(int H[], int key) {
    int index = hash(key);
    if (H[index] != 0) 
        index = probe(H, key);
    H[index] = key;
}

int search(int H[], int key) {
    int index = hash(key);
    int i = 0;
    while (H[(index + i) % n] != key) {
        if (H[(index + i) % n] == 0)
            return -1;
        i++;
        if (i == n)
            return -1;
    }
    return (index + i) % n;
}

void display(int H[]) {
    for (int i = 0; i < n; i++)
        printf("%d ", H[i]);
    printf("\n");
}

int main() {
    printf("Enter the hash table size: ");
    scanf("%d", &n);

    int* H = (int*)malloc(n * sizeof(int));
    if (!H) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    for (int i = 0; i < n; i++)
        H[i] = 0;

    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        int x;
        scanf("%d", &x);
        insert(H, x);
    }

    display(H);

    int key;
    printf("Enter the element to search: ");
    scanf("%d", &key);
    
    int index = search(H, key);
    if (index != -1)
        printf("Element found at index %d\n", index);
    else
        printf("Element not found\n");

    free(H);

    return 0;
}
