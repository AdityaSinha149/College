#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/shm.h>

typedef struct {
    char data[1024];
    int flag;
}shm;

int main() {
    int shmid = shmget(0666, sizeof(shm), 0666 | IPC_CREAT);
    if(shmid == -1) {
        perror("shmget error");
        return 1;
    }

    shm * msg = (shm *)shmat(shmid, NULL, 0);
    if(msg == (shm *)-(1)) {
        perror("shmat error");
        return 1;
    }

    msg->flag = 0;
    while(1) {
        printf("Enter your message : ");
        fgets(msg->data, 1024, stdin);
        msg->flag = 1;
        while(msg->flag == 1);
    }

    if (shmdt(msg) == -1) {
            perror("shmdt");
            exit(EXIT_FAILURE);
        }
    shmctl(shmid, IPC_RMID, NULL);
    return 0;
}