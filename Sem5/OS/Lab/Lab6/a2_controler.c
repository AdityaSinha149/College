#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>

#define FIFO_REQ "/tmp/fifo_req"
#define FIFO_RSP "/tmp/fifo_rsp"

typedef struct {
    char type;  // 'R' or 'W'
    int action; // 1 = request, 0 = release
    int pid;    // process ID
} message;

int main() {
    unlink(FIFO_REQ);
    unlink(FIFO_RSP);
    if (mkfifo(FIFO_REQ, 0666) == -1 && errno != EEXIST) { perror("mkfifo req"); exit(1); }
    if (mkfifo(FIFO_RSP, 0666) == -1 && errno != EEXIST) { perror("mkfifo rsp"); exit(1); }

    int fd_req = open(FIFO_REQ, O_RDONLY);
    int fd_rsp = open(FIFO_RSP, O_WRONLY);
    if (fd_req == -1 || fd_rsp == -1) { perror("open"); exit(1); }

    printf("Controller started...\n");

    int active_readers = 0, active_writers = 0, waiting_writers = 0;
    message msg;

    while (1) {
        if (read(fd_req, &msg, sizeof(msg)) <= 0) continue;

        if (msg.type == 'W') {
            if (msg.action == 1) { // Writer requesting
                waiting_writers++;
                while (active_readers > 0 || active_writers > 0) usleep(1000);
                waiting_writers--;
                active_writers = 1;
                write(fd_rsp, &msg.pid, sizeof(msg.pid));
            } else { // Writer releasing
                active_writers = 0;
            }
        } else if (msg.type == 'R') {
            if (msg.action == 1) { // Reader requesting
                while (active_writers > 0 || waiting_writers > 0) usleep(1000);
                active_readers++;
                write(fd_rsp, &msg.pid, sizeof(msg.pid));
            } else { // Reader releasing
                active_readers--;
            }
        }
    }

    close(fd_req);
    close(fd_rsp);
    return 0;
}
