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

    while(1) {
        while(msg->flag != 1);

        printf("%s", msg->data);
        //clear shm
        memset(msg->data, 0, 1024);
        msg->flag = 0;
    }

    if(shmdt(msg) == -1) {
        perror("shmdt error");
        return 1;
    }

    return 0;
}