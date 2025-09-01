#include <sys/wait.h>
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int pfd[2];
    if(pipe(pfd) == -1) {
        perror("pipe error");
        exit(1);
    }
    pid_t pid = fork();

    if(pid == -1) {
        perror("fork error");
        close(pfd[1]);
        close(pfd[0]);
        exit(1);
    }

    if(pid == 0) {
        //child == no right to speech
        close(pfd[1]);
        char msg[1024];
        ssize_t n = read(pfd[0], msg, sizeof(msg) - 1);
        if (n == -1) {
            perror("read error");
            exit(1);
        }
        msg[n] = '\0';
        write(1, msg, n);
        close(pfd[0]);
    }

    else {
        //parent = damaged ears|retard
        close(pfd[0]);
        char msg[1024];
        ssize_t n = read(STDIN_FILENO, msg, sizeof(msg) - 1);
        if (n == -1) {
            perror("read error");
            exit(1);
        }
        msg[n] = '\0';
        write(pfd[1], msg, n);
        close(pfd[1]);
        wait(NULL);
    }
}