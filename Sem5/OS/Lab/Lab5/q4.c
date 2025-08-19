#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main() {
    int pid = fork();

    switch (pid) {
        case -1:
            perror("fork failed");
            exit(1);

        case 0:
            printf("Child PID : %d\n", getpid());
            printf("Child exiting\n");
            exit(0);

        default:
            // Parent directly execs ps without waiting for child
            printf("Parent now running ps to show zombie...\n");
            execlp("ps", "ps", "-elf", (char *)0);

            perror("execlp failed");
            exit(1);
    }

    return 0;
}
