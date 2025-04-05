#include <stdio.h>

int knapsack(int W, int val[], int wt[], int n) {
    int dp[n][W + 1];

    for (int j = 0; j <= W; j++)
        if (j >= wt[0])
            dp[0][j] = val[0];

    for (int i = 1; i < n; i++) {
        for (int j = 0; j <= W; j++) {
            int notTake = dp[i - 1][j];
            int take = (wt[i] <= j) ? val[i] + dp[i - 1][j - wt[i]] : -1;
            dp[i][j] = (take > notTake) ? take : notTake;
        }
    }

    return dp[n - 1][W];
}

int main() {
    int n, W;
    printf("Enter number of items: ");
    scanf("%d", &n);

    int val[n], wt[n];

    printf("Enter values of items:\n");
    for (int i = 0; i < n; i++) scanf("%d", &val[i]);

    printf("Enter weights of items:\n");
    for (int i = 0; i < n; i++) scanf("%d", &wt[i]);

    printf("Enter knapsack capacity: ");
    scanf("%d", &W);

    int result = knapsack(W, val, wt, n);
    printf("Maximum value in Knapsack: %d\n", result);

    return 0;
}
