#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

void *primesInBetween(void *arg);

int isPrime(int n);

int main() {
    int n,m;
    printf("Enter start and end points : ");
    scanf("%d%d",&n,&m);
    
    n = n*10 + m;
    pthread_t tid;
    pthread_create(&tid, 0, &primesInBetween, &n);
    pthread_join(tid, NULL);


    return 0;
}

void *primesInBetween(void *arg) {
    int n = *(int*)arg;
    int m = n%10;
    n/=10;
    printf("Primes between %d and %d : ",n,m);
    for(int i = n; i <= m; i++)
        if(isPrime(i))
            printf("%d\t",i);
    printf("\n");
}

int isPrime(int n) {
    if(n == 1) return 0;
    for(int i = 2; i*i <= n; i++)
        if(n%i == 0) return 0;
    return 1;
}