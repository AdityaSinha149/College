#include <stdio.h>
#include <limits.h>

int helper(int idx, int W, int val[], int wt[], int dp[][1001]) {
    if (idx == 0)
        if (W >= wt[0]) return val[0];
        else return 0;

    if (dp[idx][W] != -1) return dp[idx][W];

    int notTake = helper(idx - 1, W, val, wt, dp);
    int take = INT_MIN;
    if (W >= wt[idx])
        take = val[idx] + helper(idx - 1, W - wt[idx], val, wt, dp);

    dp[idx][W] = (take > notTake) ? take : notTake;
    return dp[idx][W];
}

int knapsack(int W, int val[], int wt[]) {
    int n = sizeof(val[0]) ? sizeof(val) / sizeof(val[0]) : 0;
    int dp[n][W+1];

    for (int i = 0; i < n; i++)
        for (int j = 0; j <= W; j++)
            dp[i][j] = -1;

    return helper(n - 1, W, val, wt, dp);
}

int main() {
    int n, W;

    printf("Enter number of items: ");
    scanf("%d", &n);

    int val[n], wt[n];

    printf("Enter values of items:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &val[i]);
    }

    printf("Enter weights of items:\n");
    for (int i = 0; i < n; i++) {
        scanf("%d", &wt[i]);
    }

    printf("Enter knapsack capacity: ");
    scanf("%d", &W);

    int result = knapsack(W, val, wt);
    printf("Maximum value in Knapsack: %d\n", result);

    return 0;
}
