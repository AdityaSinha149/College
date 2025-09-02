#include "a2_common.h"

int main() {
    ensure_fifo("fifo_ww");

    pid_t pid = getpid();
    char reply_name[REPLY_NAME_LEN];
    snprintf(reply_name, sizeof(reply_name), "fifo_ww_%d", pid);
    ensure_fifo(reply_name);

    message req;
    memset(&req, 0, sizeof(req));
    req.pid = pid;
    req.action = 'W';
    strncpy(req.reply, reply_name, sizeof(req.reply)-1);

    int fdw = open("fifo_ww", O_WRONLY);
    write(fdw, &req, sizeof(req));
    close(fdw);
    printf("Writer %d: requested permission\n", pid);

    int fdr = open(reply_name, O_RDONLY);
    message resp;
    read(fdr, &resp, sizeof(resp));
    close(fdr);
    unlink(reply_name);

    if (resp.action == 'G') {
        printf("Writer %d: granted — writing now (20s)\n", pid);
        sleep(20);

        message rel;
        memset(&rel, 0, sizeof(rel));
        rel.pid = pid;
        rel.action = 'w';
        int fdw2 = open("fifo_ww", O_WRONLY);
        write(fdw2, &rel, sizeof(rel));
        close(fdw2);
        printf("Writer %d: released\n", pid);
    } else {
        printf("Writer %d: denied — exiting\n", pid);
    }

    return 0;
}
