#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <errno.h>

#define FIFO_REQ "/tmp/fifo_req"
#define FIFO_RSP "/tmp/fifo_rsp"

typedef struct {
    char type;  // 'R' or 'W'
    int action; // 1=request, 0=release
    int pid;
} message;

int main() {
    int fd_req = open(FIFO_REQ, O_WRONLY);
    int fd_rsp = open(FIFO_RSP, O_RDONLY);
    if (fd_req == -1 || fd_rsp == -1) { perror("open"); exit(1); }

    message msg = {'R', 1, getpid()};
    write(fd_req, &msg, sizeof(msg));

    int ack;
    read(fd_rsp, &ack, sizeof(ack));
    printf("Reader %d: Reading...\n", getpid());
    sleep(15);

    msg.action = 0;
    write(fd_req, &msg, sizeof(msg));
    printf("Reader %d: Done reading.\n", getpid());

    close(fd_req);
    close(fd_rsp);
    return 0;
}
