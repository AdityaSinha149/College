#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

int *arr;
int n;
int evenSum = 0;
int oddSum = 0;

void *sumEven(void *arg);
void *sumOdd(void *arg);

int main() {
    printf("Enter size of array: ");
    scanf("%d", &n);

    arr = (int*)malloc(n * sizeof(int));
    printf("Enter %d elements:\n", n);
    for (int i = 0; i < n; i++)
        scanf("%d", &arr[i]);

    pthread_t tid1, tid2;
    pthread_create(&tid1, 0, &sumEven, NULL);
    pthread_create(&tid2, 0, &sumOdd, NULL);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    printf("Sum of even numbers = %d\n", evenSum);
    printf("Sum of odd numbers  = %d\n", oddSum);

    free(arr);
    return 0;
}

void *sumEven(void *arg) {
    for (int i = 0; i < n; i++) {
        if (arr[i] % 2 == 0)
            evenSum += arr[i];
    }
    return NULL;
}

void *sumOdd(void *arg) {
    for (int i = 0; i < n; i++) {
        if (arr[i] % 2 != 0)
            oddSum += arr[i];
    }
    return NULL;
}
