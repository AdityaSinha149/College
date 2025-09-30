//q3.c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

sem_t A,B;

void *t1(void *p){
    sem_wait(&A);
    printf("T1 got A\n");
    sleep(1);
    sem_wait(&B);
    printf("T1 got B\n");
    return 0;
}

void *t2(void *p){
    sem_wait(&B);
    printf("T2 got B\n");
    sleep(1);
    sem_wait(&A);
    printf("T2 got A\n");
    return 0;
}

int main(){
    pthread_t x,y;
    sem_init(&A,0,1); sem_init(&B,0,1);
    pthread_create(&x,0,t1,0);
    pthread_create(&y,0,t2,0);
    pthread_join(x,0);
    pthread_join(y,0);
}
