#include "banker.h"
#include <unistd.h>

void* process_thread(void* arg) {
    int p = *((int*)arg);
    int req[R];

    sleep(1 + rand() % 3);
    for (int i = 0; i < R; i++)
        req[i] = rand() % (need[p][i] + 1);

    if (request_resources(p, req) == 0) {
        sleep(2 + rand() % 3);
        release_resources(p, req);
    }

    printf("Process %d finished.\n", p);
    pthread_exit(NULL);
}

int main() {
    srand(time(NULL));
    pthread_t threads[P];
    int process_ids[P];

    pthread_mutex_init(&lock, NULL);
    calculate_need();

    for (int i = 0; i < P; i++) {
        process_ids[i] = i;
        pthread_create(&threads[i], NULL, process_thread, &process_ids[i]);
    }

    for (int i = 0; i < P; i++)
        pthread_join(threads[i], NULL);

    pthread_mutex_destroy(&lock);
    printf("\nAll processes completed.\n");
    return 0;
}
