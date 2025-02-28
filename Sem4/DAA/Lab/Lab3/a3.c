#include <stdio.h>

int isPartitionPossible(int arr[], int n, int sum) {
    if (sum == 0) return 1;
    if (n == 0 || sum < 0) return 0;
    return isPartitionPossible(arr, n - 1, sum - arr[n - 1]) || isPartitionPossible(arr, n - 1, sum);
}

int canPartition(int arr[], int n) {
    int sum = 0;
    for (int i = 0; i < n; i++)
        sum += arr[i];

    if (sum % 2 != 0)
        return 0;

    return isPartitionPossible(arr, n, sum / 2);
}

int main() {
    int n;
    printf("Enter number of elements: ");
    scanf("%d", &n);

    int arr[n];
    printf("Enter elements: ");
    for (int i = 0; i < n; i++)
        scanf("%d", &arr[i]);

    if (canPartition(arr, n))
        printf("Partition possible\n");
    else
        printf("Partition not possible\n");

    return 0;
}
