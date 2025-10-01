#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <unistd.h>
#include <semaphore.h>
#include <time.h>

// Shared semaphores
sem_t mutex;        // Protects read_count
sem_t rw_mutex;     // Ensures writer exclusivity
sem_t queue_mutex;  // Protects writer_waiting

int read_count = 0;       // Active readers
int writer_waiting = 0;   // Writers waiting in queue

// ---------------- Reader ----------------
void *reader(void *arg) {
    int id = *(int *)arg;
    free(arg);

    // Check if writer is waiting
    sem_wait(&queue_mutex);
    if (writer_waiting > 0) {
        printf("[Reader %d] Declined (writer waiting)\n", id);
        sem_post(&queue_mutex);
        return NULL;
    }
    sem_post(&queue_mutex);

    // Entry section
    sem_wait(&mutex);
    read_count++;
    if (read_count == 1)
        sem_wait(&rw_mutex);
    sem_post(&mutex);

    // Reading
    printf("[Reader %d] Reading...\n", id);
    sleep(2);
    printf("[Reader %d] Finished Reading\n", id);

    // Exit section
    sem_wait(&mutex);
    read_count--;
    if (read_count == 0)
        sem_post(&rw_mutex);
    sem_post(&mutex);

    return NULL;
}

/// ---------------- Writer ----------------
void *writer(void *arg) {
    int id = *(int *)arg;
    free(arg);

    // Indicate writer wants to enter
    sem_wait(&queue_mutex);
    writer_waiting++;
    printf("[Writer %d] Joined writing queue (waiting writers = %d)\n", id, writer_waiting);
    sem_post(&queue_mutex);

    // Entry section
    sem_wait(&rw_mutex);

    // Writing
    printf("[Writer %d] Writing...\n", id);
    sleep(1);
    printf("[Writer %d] Finished Writing\n", id);

    // Done
    sem_wait(&queue_mutex);
    writer_waiting--;
    sem_post(&queue_mutex);

    sem_post(&rw_mutex);

    return NULL;
}

// ---------------- Manager Thread ----------------
void *manager(void *arg) {
    int i = 0;
    while (1) {
        pthread_t tid;
        int *id = malloc(sizeof(int));
        *id = ++i;

        int choice = rand() % 2; // 0 = reader, 1 = writer
        if (choice == 0) {
            pthread_create(&tid, NULL, reader, id);
            pthread_detach(tid); // auto cleanup
        } else {
            if(rand()%2 == 1) continue;
            pthread_create(&tid, NULL, writer, id);
            pthread_detach(tid);
        }

        sleep(rand() % 3 + 1); // Wait 1–3 sec before next thread
    }
    return NULL;
}

// ---------------- Main ----------------
int main() {
    srand(time(NULL));

    // Init semaphores
    sem_init(&mutex, 0, 1);
    sem_init(&rw_mutex, 0, 1);
    sem_init(&queue_mutex, 0, 1);

    pthread_t mgr;
    pthread_create(&mgr, NULL, manager, NULL);
    pthread_join(mgr, NULL);

    // Cleanup (never reached)
    sem_destroy(&mutex);
    sem_destroy(&rw_mutex);
    sem_destroy(&queue_mutex);

    return 0;
}
