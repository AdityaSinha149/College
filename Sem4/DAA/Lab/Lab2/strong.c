#include <stdio.h>

int factorial(int n) {
    if (n == 0 || n == 1) {
        return 1;
    }
    return n * factorial(n - 1);
}

int isStrong(int num) {
    int sum = 0, original = num;
    while (num > 0) {
        int digit = num % 10;
        sum += factorial(digit);
        num /= 10;
    }
    return sum == original;
}

void deleteStrongNumbers(int arr[], int *size, int index) {
    if (index >= *size) {
        return;
    }
    if (isStrong(arr[index])) {
        for (int i = index; i < *size - 1; i++) {
            arr[i] = arr[i + 1];
        }
        (*size)--;
        deleteStrongNumbers(arr, size, index);
    } else {
        deleteStrongNumbers(arr, size, index + 1);
    }
}

int main() {
    int arr[] = {1, 145, 3, 4, 5};
    int size = sizeof(arr) / sizeof(arr[0]);
    deleteStrongNumbers(arr, &size, 0);
    printf("Array after deleting strong numbers: ");
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
    return 0;
}