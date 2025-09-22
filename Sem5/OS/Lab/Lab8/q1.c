#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
#include <math.h>

void *produce(void *arg);
void *consume(void *arg);

sem_t mutex;
int produced = 0, consumed = 0;

void *produce(void *arg) {
    while (1) {
        while (produced - consumed >= 10);
        sem_wait(&mutex);
        produced++;
        printf("[producer] : produced item %d\n", produced);
        printf("No of products available : %d\n", produced - consumed);
        sem_post(&mutex);

        sleep(rand()%10);
    }
}

void *consume(void *arg) {
    while (1) {
        while (consumed == produced);
        sem_wait(&mutex);
        consumed++;
        printf("[consumer] : consumed item %d\n", consumed);
        sem_post(&mutex);

        sleep(rand()%10);
    }
}

int main() {
    pthread_t tid1, tid2;
    srand(time(NULL));
    sem_init(&mutex, 0, 1);

    pthread_create(&tid1, NULL, produce, NULL);
    pthread_create(&tid2, NULL, consume, NULL);

    pthread_join(tid1, NULL);
    pthread_join(tid2, NULL);

    return 0;
}
