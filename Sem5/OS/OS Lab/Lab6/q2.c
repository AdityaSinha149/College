#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>
#include <sys/types.h>

int main() {
    int fd[2];  // fd[0] for reading, fd[1] for writing
    pid_t pid;
    char write_msg[100];
    char read_msg[100];

    // create a pipe
    if (pipe(fd) == -1) {
        perror("pipe");
        exit(1);
    }

    pid = fork();
    if (pid < 0) {
        perror("fork");
        exit(1);
    }

    if (pid > 0) {
        // Parent doesnt listen only instructs close reading end
        close(fd[0]);

        printf("Enter a message to send to child: ");
        fgets(write_msg, sizeof(write_msg), stdin);

        write(fd[1], write_msg, strlen(write_msg) + 1);
        close(fd[1]);
    } else {
        // Child doesnt question close writing end
        close(fd[1]);
        read(fd[0], read_msg, sizeof(read_msg));
        printf("Child received: %s\n", read_msg);
        close(fd[0]);
    }

    return 0;
}
