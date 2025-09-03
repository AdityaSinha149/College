#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/shm.h>
#include <sys/types.h>

#define SHM_SIZE 1024

typedef struct {
    char data[SHM_SIZE];
    int flag; // 0 = empty, 1 = written by parent, 2 = written by child
} shared_mem;

int main() {
    int shmid = shmget(IPC_PRIVATE, sizeof(shared_mem), 0666 | IPC_CREAT);
    if (shmid == -1) {
        perror("shmget");
        exit(EXIT_FAILURE);
    }

    shared_mem *shm = (shared_mem *)shmat(shmid, NULL, 0);
    if (shm == (shared_mem *)(-1)) {
        perror("shmat");
        exit(EXIT_FAILURE);
    }

    shm->flag = 0; // Start empty

    pid_t pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(EXIT_FAILURE);
    }

    if (pid > 0) {
        // PARENT PROCESS
        printf("Parent: Enter an alphabet: ");
        if (fgets(shm->data, SHM_SIZE, stdin) == NULL) {
            perror("fgets");
            exit(EXIT_FAILURE);
        }
        shm->flag = 1; // Mark as written

        while (shm->flag != 2);

        printf("Parent: Child sent back: %s", shm->data);

        if (shmdt(shm) == -1) {
            perror("shmdt");
            exit(EXIT_FAILURE);
        }
        shmctl(shmid, IPC_RMID, NULL);
    } 
    else {
        // CHILD PROCESS
        // Wait for parent
        while (shm->flag != 1) {
            usleep(1000);
        }

        printf("Child: Received '%c'\n", shm->data[0]);
        char c = shm->data[0];
        c++;
        if (c == 'z' + 1 || c == 'Z' + 1)
            c -= 26;
        shm->data[0] = c;

        shm->flag = 2; // Mark as processed

        if (shmdt(shm) == -1) {
            perror("shmdt");
            exit(EXIT_FAILURE);
        }
    }

    return 0;
}
