#include <stdio.h>
#include <stdlib.h>

void swap(int* arr, int i, int j) {
    arr[i] ^= arr[j];
    arr[j] ^= arr[i];
    arr[i] ^= arr[j];
}

void bubbleSort(int* arr, int n) {
    for (int i = 0; i < n - 1; i++) {
        int done = 1;
        for (int j = 0; j < n - 1 - i; j++)
            if (arr[j] > arr[j + 1]) {
                swap(arr, j, j + 1);
                done = 0;
            }
        if (done == 1) return;
    }
}

int main() {
    int n;
    printf("Enter the array length: ");
    scanf("%d", &n);
    int* arr = (int*)malloc(n * sizeof(int));

    printf("Enter %d elements: \n", n);
    for (int i = 0; i < n; i++)
        scanf("%d", arr + i);
    
    printf("Before sort: ");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    bubbleSort(arr, n);

    printf("\nAfter sort: ");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    free(arr);
    return 0;
}
