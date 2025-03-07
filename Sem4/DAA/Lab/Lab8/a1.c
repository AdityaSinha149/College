#include <stdio.h>
#include <stdlib.h>

int heapify(int arr[], int n, int i) {
    int left = 2 * i + 1;
    int right = 2 * i + 2;

    if (left < n && arr[left] > arr[i])
        return 0;
    if (right < n && arr[right] > arr[i])
        return 0;

    return 1;
}

int checkHeap(int arr[], int n) {
    for (int i = n / 2 - 1; i >= 0; i--)
        if (heapify(arr, n, i) == 0)
            return 0; 
    return 1;
}

int main() {
    int n;
    printf("Enter the number of elements: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter the elements: ");
    for (int i = 0; i < n; i++)
        scanf("%d", &arr[i]);

    if (checkHeap(arr, n))
        printf("The array is a valid max heap.\n");
    else
        printf("The array is not a valid max heap.\n");

    return 0;
}
