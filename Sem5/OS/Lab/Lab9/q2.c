#include <stdio.h>
#include <pthread.h>
#include <stdbool.h>

int number = 0;
int sum = 0;
bool ready = false;
bool done = false;

void* adder(void* arg) {
    while (!done) {
        if (ready) {
            sum += number;
            ready = false;
        }
    }
    return NULL;
}

int main() {
    pthread_t tid;
    pthread_create(&tid, NULL, adder, NULL);

    while (1) {
        int x;
        printf("Enter non-negative int (-1 to exit): ");
        scanf("%d", &x);

        if (x < 0) {
            done = true;
            break;
        }

        number = x;
        ready = true;
        while (ready);
    }

    pthread_join(tid, NULL);
    printf("Final sum = %d\n", sum);
    return 0;
}
