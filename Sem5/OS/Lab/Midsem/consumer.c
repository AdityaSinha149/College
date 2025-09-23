#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <time.h>
#include <unistd.h>
#include <errno.h>

#define NUM_PRODUCERS 4
#define ITEMS_PER_PRODUCER 5
#define QUEUE_KEY 1234

struct msgbuf {
    long mtype;
    int product_value;
};

int main() {
    int msgid;
    struct msgbuf message;

    srand(time(NULL) ^ getpid());

    msgid = msgget(QUEUE_KEY, 0666);
    if (msgid == -1) {
        perror("msgget");
        exit(EXIT_FAILURE);
    }

    int total_products = NUM_PRODUCERS * ITEMS_PER_PRODUCER;
    int consumed = 0;

    while (consumed < total_products) {
        int chosen_producer = (rand() % NUM_PRODUCERS) + 1;

        int ret = msgrcv(msgid, &message, sizeof(int), chosen_producer, IPC_NOWAIT);

        if (ret == -1) {
                continue;

        } else {
            printf("Consumer consumed product value: %d from Producer-%ld\n", message.product_value, message.mtype);
            consumed++;
        }

        sleep(5);
    }

    printf("Consumer has consumed all products. Exiting...\n");

    msgctl(msgid, IPC_RMID, NULL);

    return 0;
}