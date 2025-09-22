#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>

void *produce(void *arg);

void *consume(void *arg);

sem_t mutex;
int slots, full;
int i,j;

int main() {
    pthread_t tid1, tid2;
    sem_init(&mutex, 0, 1);
    pthread_create(&tid1,NULL,produce,NULL);
    pthread_create(&tid2,NULL,consume,NULL);
    pthread_join(tid1,NULL);
    pthread_join(tid2,NULL);
}

void *produce(void *arg) {
    while(full == slots);
    sem_wait(&mutex);
    full++;
    printf("[producer] : produced item %d", i++);
    sleep(2);
    sem_post(&mutex);
}

void *consume(void *arg) {
    while(full == 0);
    sem_wait(&mutex);
    full--;
    sem_post(slots);
    sem_wait(full);
    printf("[consumer] : consumed item %d", i++);
    sleep(2);
    sem_post(&mutex);
}