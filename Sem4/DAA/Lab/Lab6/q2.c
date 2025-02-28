#include <stdio.h>
#include <stdlib.h>

int count = 0;

void swap(int *a, int *b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int Partition(int *arr, int l, int r) {
    int p = arr[l];
    int i = l;
    int j = r;

    while (i <= j) {
        while (i <= r && arr[i] <= p) {
            i++;
            count++;
        }
        while (arr[j] > p && j >= l) {
            j--;
            count++;
        }
        if (i < j) {
            swap(&arr[i], &arr[j]);
        }
    }
    count++;
    swap(&arr[l], &arr[j]);
    return j;
}

void QuickSort(int *arr, int l, int r) {
    if (l >= r) return;
    int p = Partition(arr, l, r);
    QuickSort(arr, l, p - 1);
    QuickSort(arr, p + 1, r);
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);
    
    int *arr = (int *)malloc(n * sizeof(int));
    if (arr == NULL) {
        printf("Memory allocation failed.\n");
        return 1;
    }

    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    QuickSort(arr, 0, n - 1);

    printf("Sorted array: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");

    printf("Number of comparisons: %d\n", count);

    free(arr);
    return 0;
}
