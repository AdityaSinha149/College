#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>

#define CHAIRS 3   // number of waiting chairs

sem_t customers;   // counts waiting customers
sem_t barbers;     // counts barber's availability
pthread_mutex_t mutex;  // for mutual exclusion

int waiting = 0;   // count of waiting customers

void *barber(void *arg) {
    while (1) {
        // wait for a customer
        sem_wait(&customers);

        // barber is ready
        pthread_mutex_lock(&mutex);
        waiting--;
        printf("Barber is cutting hair... (waiting customers: %d)\n", waiting);
        pthread_mutex_unlock(&mutex);

        sem_post(&barbers); // barber is now ready
        sleep(5); // simulate haircut
    }
}

void *customer(void *arg) {
    pthread_mutex_lock(&mutex);
    if (waiting < CHAIRS) {
        waiting++;
        printf("Customer came and is waiting... (waiting customers: %d)\n", waiting);
        pthread_mutex_unlock(&mutex);

        sem_post(&customers); // notify barber
        sem_wait(&barbers);   // wait for barber
        printf("Customer is getting a haircut.\n");
    } else {
        // no chair, leave shop
        printf("Customer left (no chair available).\n");
        pthread_mutex_unlock(&mutex);
    }
    return NULL;
}

int main() {
    pthread_t barberThread;
    pthread_t custThread;
    int id = 1;

    srand(time(NULL));

    sem_init(&customers, 0, 0);
    sem_init(&barbers, 0, 0);
    pthread_mutex_init(&mutex, NULL);

    pthread_create(&barberThread, NULL, barber, NULL);

    // customers keep arriving randomly
    while (1) {
        sleep(rand() % 3 + 1);  // random interval between arrivals

        pthread_create(&custThread, NULL, customer, NULL);
        pthread_detach(custThread); // auto-cleanup, no need for join

        id++;
    }

    pthread_join(barberThread, NULL);
    return 0;
}
