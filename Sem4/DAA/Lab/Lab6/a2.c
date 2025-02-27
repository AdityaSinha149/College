#include <stdio.h>

void multiply(int F[2][2], int M[2][2]) {
    int x = F[0][0] * M[0][0] + F[0][1] * M[1][0];
    int y = F[0][0] * M[0][1] + F[0][1] * M[1][1];
    int z = F[1][0] * M[0][0] + F[1][1] * M[1][0];
    int w = F[1][0] * M[0][1] + F[1][1] * M[1][1];

    F[0][0] = x;
    F[0][1] = y;
    F[1][0] = z;
    F[1][1] = w;
}

void power(int F[2][2], int n) {
    if (n == 0 || n == 1) return;

    int M[2][2] = {{1, 1}, {1, 0}};

    power(F, n / 2);
    multiply(F, F);

    if (n % 2 != 0) multiply(F, M);
}

int countWays(int n) {
    if (n == 1) return 1;
    if (n == 2) return 2;

    int F[2][2] = {{1, 1}, {1, 0}};
    power(F, n);
    return F[0][0];
}

int main() {
    int n;
    printf("Enter the value of n: ");
    scanf("%d", &n);
    printf("Number of ways to tile 2 x %d board: %d\n", n, countWays(n));
    return 0;
}
