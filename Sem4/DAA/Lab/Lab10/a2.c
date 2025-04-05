#include <stdio.h>

void findSubset(int W, int wt[], int val[], int n, int dp[n][W+1]) {
    int w = W;
    printf("Items included in the knapsack:\n");

    for (int i = n - 1; i > 0; i--) {
        if (dp[i][w] != dp[i - 1][w]) {
            printf("Item %d (value=%d, weight=%d)\n", i, val[i], wt[i]);
            w -= wt[i];
        }
    }

    if (w >= wt[0] && dp[0][w] != 0) {
        printf("Item 0 (value=%d, weight=%d)\n", val[0], wt[0]);
    }
}
