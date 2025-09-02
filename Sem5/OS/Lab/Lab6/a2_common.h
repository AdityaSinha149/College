#ifndef A2_COMMON_H
#define A2_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <string.h>
#include <errno.h>

#define REPLY_NAME_LEN 64

typedef struct {
    pid_t pid;
    char action;                // 'R' request read, 'r' release read
                                // 'W' request write, 'w' release write
                                // 'G' grant, 'D' deny
                                // 'S' sync (controller->controller)
    char reply[REPLY_NAME_LEN]; // client reply fifo name (set by client for controller reply)
    int readers;                // used by S messages
    int waiting_writers;        // used by S messages
    int writer_active;          // used by S messages (0/1)
} message;

static inline void ensure_fifo(const char *path) {
    if (access(path, F_OK) == -1) {
        if (mkfifo(path, 0666) == -1 && errno != EEXIST) {
            perror("mkfifo");
            exit(1);
        }
    }
}

#endif
