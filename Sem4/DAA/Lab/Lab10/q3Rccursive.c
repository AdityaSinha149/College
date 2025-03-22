#include <stdio.h>

int max(int a, int b) {
    return (a > b) ? a : b;
}

int knapsackRecursive(int weights[], int values[], int n, int W, int** dp) {
    if (n == 0 || W == 0)
        return 0;
    if (dp[n][W] != -1)
        return dp[n][W];

    if (weights[n - 1] > W) {
        dp[n][W] = knapsackRecursive(weights, values, n - 1, W, dp);
    } else {
        dp[n][W] = max(values[n - 1] + knapsackRecursive(weights, values, n - 1, W - weights[n - 1], dp),
                       knapsackRecursive(weights, values, n - 1, W, dp));
    }

    return dp[n][W];
}

int knapsack(int weights[], int values[], int n, int W) {
    int** dp = (int**)malloc((n + 1) * sizeof(int*));
    for (int i = 0; i <= n; i++) {
        dp[i] = (int*)malloc((W + 1) * sizeof(int));
        for (int j = 0; j <= W; j++) {
            dp[i][j] = -1;
        }
    }
    int result = knapsackRecursive(weights, values, n, W, dp);
    return result;
}

int main() {
    int n, W;
    printf("Enter the number of items: ");
    scanf("%d", &n);
    int weights[n], values[n];
    printf("Enter the weights of the items:\n");
    for (int i = 0; i < n; i++)
        scanf("%d", &weights[i]);
    printf("Enter the values of the items:\n");
    for (int i = 0; i < n; i++)
        scanf("%d", &values[i]);
    printf("Enter the maximum weight capacity of the knapsack: ");
    scanf("%d", &W);
    int maxValue = knapsack(weights, values, n, W);
    printf("The maximum value that can be obtained is: %d\n", maxValue);
    return 0;
}