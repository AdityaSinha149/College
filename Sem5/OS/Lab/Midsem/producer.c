#include <stdio.h>
#include <stdlib.h>
#include <sys/ipc.h>
#include <sys/msg.h>
#include <time.h>
#include <unistd.h>

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

    msgid = msgget(QUEUE_KEY, IPC_CREAT | 0666);
    if (msgid == -1) {
        perror("msgget");
        exit(EXIT_FAILURE);
    }

    int total_products = NUM_PRODUCERS * ITEMS_PER_PRODUCER;

    for (int i = 0; i < total_products; i++) {
        int producer_id = (rand() % NUM_PRODUCERS) + 1;
        int product_value = rand() % 100;

        message.mtype = producer_id;
        message.product_value = product_value;

        if (msgsnd(msgid, &message, sizeof(int), 0) == -1) {
            perror("msgsnd");
            exit(EXIT_FAILURE);
        }

        printf("Producer-%d produced product value: %d\n", producer_id, product_value);

        sleep(2);
    }

    return 0;
}
