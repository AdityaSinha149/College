//q2.c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

sem_t wrt;
pthread_mutex_t mutex;
int readcount = 0, data = 0;

void *reader(void *arg) {
    int id = *(int*)arg;
    while (1) {
        pthread_mutex_lock(&mutex);
        if (++readcount == 1) sem_wait(&wrt);
        pthread_mutex_unlock(&mutex);

        printf("Reader %d read: %d\n", id, data);

        pthread_mutex_lock(&mutex);
        if (--readcount == 0) sem_post(&wrt);
        pthread_mutex_unlock(&mutex);

        sleep(1);
    }
}

void *writer(void *arg) {
    int id = *(int*)arg;
    while (1) {
        sem_wait(&wrt);
        printf("Writer %d wrote: %d\n", id, ++data);
        sem_post(&wrt);
        sleep(2);
    }
}

int main() {
    pthread_t r[3], w[2];
    int id[5];
    sem_init(&wrt,0,1);
    pthread_mutex_init(&mutex,0);

    for(int i=0;i<3;i++){ id[i]=i+1; pthread_create(&r[i],0,reader,&id[i]); }
    for(int i=0;i<2;i++){ id[3+i]=i+1; pthread_create(&w[i],0,writer,&id[3+i]); }

    for(int i=0;i<3;i++) pthread_join(r[i],0);
    for(int i=0;i<2;i++) pthread_join(w[i],0);
}
