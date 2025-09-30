//q4
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define CHAIRS 3

sem_t customers, barbers;
pthread_mutex_t mutex;
int waiting = 0;

void *barber(void *arg) {
    while (1) {
        sem_wait(&customers);             // wait for customer
        pthread_mutex_lock(&mutex);
        waiting--;                        // one customer goes to barber
        pthread_mutex_unlock(&mutex);
        sem_post(&barbers);               // barber ready
        printf("Barber cutting hair\n");
        sleep(2);                         // haircut time
    }
}

void *customer(void *arg) {
    int id = *(int*)arg;
    pthread_mutex_lock(&mutex);
    if (waiting < CHAIRS) {
        waiting++;
        printf("Customer %d waiting\n", id);
        pthread_mutex_unlock(&mutex);
        sem_post(&customers);             // notify barber
        sem_wait(&barbers);               // wait for barber
        printf("Customer %d getting haircut\n", id);
    } else {
        printf("Customer %d left (no chair)\n", id);
        pthread_mutex_unlock(&mutex);
    }
    return 0;
}

int main() {
    pthread_t b, c[5];
    int id[5];
    sem_init(&customers,0,0);
    sem_init(&barbers,0,0);
    pthread_mutex_init(&mutex,0);

    pthread_create(&b,0,barber,0);
    for(int i=0;i<5;i++){
        id[i]=i+1;
        pthread_create(&c[i],0,customer,&id[i]);
        sleep(1); // customers arriving at different times
    }
    for(int i=0;i<5;i++) pthread_join(c[i],0);
}
