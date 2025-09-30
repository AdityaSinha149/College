#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *fibonacci(void *arg);

int *arr;

int main() {
    int n;
    printf("Enter the number of fibonnaci numbers : ");
    scanf("%d",&n);

    arr = (int*)malloc(n*sizeof(int));
    pthread_t tid;
    pthread_create(&tid, 0, &fibonacci, &n);
    pthread_join(tid, NULL);

    printf("Ans by child : ");
    for(int i = 0; i < n; i++)
        printf("%d\t",arr[i]);
    return 0;
}

void *fibonacci(void *arg) {
    int n = *(int*)arg;
    arr[0]=0;
    arr[1]=1;
    for(int i = 2; i < n; i++)
        arr[i] = arr[i - 1] + arr[i - 2];
}