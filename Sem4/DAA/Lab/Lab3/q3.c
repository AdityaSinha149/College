#include <stdio.h>

int canPartition(int arr[], int n, int sum, int currentSum, int index) {
    if (currentSum == sum) return 1;
    if (index == n || currentSum > sum) return 0;
    return canPartition(arr, n, sum, currentSum + arr[index], index + 1)
     ||canPartition(arr, n, sum, currentSum, index + 1);
}

int isPartitionPossible(int arr[], int n) {
    int totalSum = 0;
    for (int i = 0; i < n; i++) totalSum += arr[i];
    if (totalSum % 2 != 0) return 0;
    return canPartition(arr, n, totalSum / 2, 0, 0);
}

int main() {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);
    
    int arr[n];
    printf("Enter the elements: ");
    for (int i = 0; i < n; i++) {
        scanf("%d", &arr[i]);
    }

    if (isPartitionPossible(arr, n)) {
        printf("Partition is possible.\n");
    } else {
        printf("Partition is not possible.\n");
    }
    
    return 0;
}
