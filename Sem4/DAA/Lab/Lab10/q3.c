#include <stdio.h>
int knapsack(int weights[], int values[], int n, int W) {
    int dp[1000] = {0}; 
    int prev[1000] = {0};

    for (int i = 0; i < n; i++) {
        for (int w = 0; w <= W; w++)
            prev[w] = dp[w];

        for (int w = W; w >= weights[i]; w--) {
            dp[w] = prev[w];
            if (prev[w - weights[i]] + values[i] > dp[w])
                dp[w] = prev[w - weights[i]] + values[i]; 
        }
    }
    return dp[W];
}

void main() {
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
}