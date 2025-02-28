#include <stdio.h>

int mypow(int a, int n) {
    if (n == 0) return 1;
    if (n == 1) return a;
    int half = mypow(a, n / 2);
    return (n % 2 == 0) ? half * half : a * half * half;
}

int main() {
    int a, n;
    printf("Enter base and exponent: ");
    scanf("%d%d", &a, &n);
    printf("Result: %d", mypow(a, n));
    return 0;
}
