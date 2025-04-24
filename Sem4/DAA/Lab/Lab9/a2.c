#include <stdio.h>

void distributionCountingSort(int arr[], int n) {
    int i, min = arr[0], max = arr[0];

    // Find the min and max in the array
    for (i = 1; i < n; i++) {
        if (arr[i] < min) min = arr[i];
        if (arr[i] > max) max = arr[i];
    }

    int range = max - min + 1;
    int count[range];
    int output[n];

    // Initialize count array
    for (i = 0; i < range; i++)
        count[i] = 0;

    // Count occurrences
    for (i = 0; i < n; i++)
        count[arr[i] - min]++;

    // Compute distribution count (cumulative sum)
    for (i = 1; i < range; i++)
        count[i] += count[i - 1];

    // Place elements in output array using distribution count
    for (i = n - 1; i >= 0; i--) {
        output[count[arr[i] - min] - 1] = arr[i];
        count[arr[i] - min]--;
    }

    // Copy sorted output to original array
    for (i = 0; i < n; i++)
        arr[i] = output[i];
}

int main() {
    int arr[100], n;

    printf("Enter number of elements: ");
    scanf("%d", &n);

    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++)
        scanf("%d", &arr[i]);

    distributionCountingSort(arr, n);

    printf("Sorted array:\n");
    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);

    return 0;
}
