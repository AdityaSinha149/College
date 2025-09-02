#include "a2_common.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <string.h>

int main() {
    ensure_fifo("fifo_rr");
    ensure_fifo("fifo_c");

    pid_t pid = getpid();
    char reply_name[REPLY_NAME_LEN];
    snprintf(reply_name, sizeof(reply_name), "/tmp/fifo_rr_%d", pid);
    ensure_fifo(reply_name);

    message req;
    memset(&req,0,sizeof(req));
    req.pid = pid;
    req.action = 'R';
    strncpy(req.reply, reply_name, sizeof(req.reply)-1);

    int fdw = open("/tmp/fifo_rr", O_WRONLY);
    if(fdw == -1) { perror("open fifo_rr"); unlink(reply_name); return 1; }
    write(fdw, &req, sizeof(req));
    close(fdw);
    printf("Reader %d: requested permission\n", pid);

    int fdr = open(reply_name, O_RDONLY);
    if(fdr == -1) { perror("open reply fifo"); unlink(reply_name); return 1; }
    message resp;
    if(read(fdr, &resp, sizeof(resp)) != sizeof(resp)) { perror("read resp"); close(fdr); unlink(reply_name); return 1; }
    close(fdr);
    unlink(reply_name);

    if(resp.action == 'G') {
        printf("Reader %d: granted — reading now (20s)\n", pid);
        sleep(20);

        message rel;
        memset(&rel,0,sizeof(rel));
        rel.pid = pid;
        rel.action = 'r';
        int fdw2 = open("/tmp/fifo_rr", O_WRONLY);
        if(fdw2 != -1) { write(fdw2, &rel, sizeof(rel)); close(fdw2); }
        printf("Reader %d: released\n", pid);
    } else {
        printf("Reader %d: denied (writer active or waiting)\n", pid);
    }

    return 0;
}
