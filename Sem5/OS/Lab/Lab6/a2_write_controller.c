#include "a2_common.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/select.h>
#include <string.h>
#include <errno.h>

#define MAX_WAITING_WRITERS 128

typedef struct {
    pid_t pid;
    char reply[REPLY_NAME_LEN];
} writer_req_t;

int main() {
    ensure_fifo("fifo_ww");          // Writer requests
    ensure_fifo("fifo_rc_to_wc");    // RC -> WC
    ensure_fifo("fifo_wc_to_rc");    // WC -> RC

    int local_writer_active = 0;
    int remote_readers = 0;          // readers from RC

    writer_req_t waiting_queue[MAX_WAITING_WRITERS];
    int queue_start = 0, queue_end = 0; // circular queue

    printf("Writer Controller started (writer-priority)\n");

    int fifo_ww_fd = open("fifo_ww", O_RDONLY | O_NONBLOCK);
    int fifo_rc_fd = open("fifo_rc_to_wc", O_RDWR | O_NONBLOCK); // O_RDWR prevents ENXIO
    int fifo_wc_to_rc_fd = open("fifo_wc_to_rc", O_RDWR | O_NONBLOCK);

    if (fifo_ww_fd == -1 || fifo_rc_fd == -1 || fifo_wc_to_rc_fd == -1) {
        perror("open fifo");
        return 1;
    }

    while (1) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fifo_ww_fd, &rfds);
        FD_SET(fifo_rc_fd, &rfds);
        int maxfd = (fifo_ww_fd > fifo_rc_fd ? fifo_ww_fd : fifo_rc_fd) + 1;

        if (select(maxfd, &rfds, NULL, NULL, NULL) <= 0) continue;

        // 1) Process RC sync messages
        if (FD_ISSET(fifo_rc_fd, &rfds)) {
            message sync;
            while (read(fifo_rc_fd, &sync, sizeof(sync)) == sizeof(sync)) {
                if (sync.action == 'S') {
                    remote_readers = sync.readers;
                    printf("WC: received sync -> remote_readers=%d local_writer_active=%d waiting_in_queue=%d\n",
                           remote_readers, local_writer_active, (queue_end - queue_start + MAX_WAITING_WRITERS) % MAX_WAITING_WRITERS);
                }
            }
        }

        // 2) Process writer requests
        if (FD_ISSET(fifo_ww_fd, &rfds)) {
            message req;
            while (read(fifo_ww_fd, &req, sizeof(req)) == sizeof(req)) {
                if (req.action == 'W') {
                    // Enqueue writer request
                    int next_end = (queue_end + 1) % MAX_WAITING_WRITERS;
                    if (next_end == queue_start) {
                        printf("WC: waiting queue full, cannot enqueue writer %d\n", req.pid);
                        continue;
                    }
                    waiting_queue[queue_end].pid = req.pid;
                    strncpy(waiting_queue[queue_end].reply, req.reply, REPLY_NAME_LEN);
                    queue_end = next_end;
                    printf("WC: writer %d enqueued (queue_size=%d)\n",
                           req.pid, (queue_end - queue_start + MAX_WAITING_WRITERS) % MAX_WAITING_WRITERS);

                    // Send sync to RC: writer waiting
                    message sync;
                    memset(&sync, 0, sizeof(sync));
                    sync.action = 'S';
                    sync.readers = remote_readers;
                    sync.waiting_writers = (queue_end - queue_start + MAX_WAITING_WRITERS) % MAX_WAITING_WRITERS;
                    sync.writer_active = local_writer_active;
                    write(fifo_wc_to_rc_fd, &sync, sizeof(sync));
                    printf("WC: sent sync -> writer_active=%d waiting_writers=%d readers=%d\n",
                           local_writer_active, sync.waiting_writers, remote_readers);
                } else if (req.action == 'w') {
                    // Current writer finished
                    local_writer_active = 0;
                    printf("WC: writer %d finished\n", req.pid);

                    // Send sync to RC
                    message sync;
                    memset(&sync, 0, sizeof(sync));
                    sync.action = 'S';
                    sync.readers = remote_readers;
                    sync.waiting_writers = (queue_end - queue_start + MAX_WAITING_WRITERS) % MAX_WAITING_WRITERS;
                    sync.writer_active = local_writer_active;
                    write(fifo_wc_to_rc_fd, &sync, sizeof(sync));
                    printf("WC: sent sync -> writer_active=%d waiting_writers=%d readers=%d\n",
                           local_writer_active, sync.waiting_writers, remote_readers);
                }
            }
        }

        // 3) Grant next waiting writer if possible
        if (!local_writer_active && remote_readers == 0 &&
            queue_start != queue_end) { // queue not empty
            writer_req_t next = waiting_queue[queue_start];
            queue_start = (queue_start + 1) % MAX_WAITING_WRITERS;

            // Grant writer
            local_writer_active = 1;
            message resp;
            memset(&resp, 0, sizeof(resp));
            resp.pid = next.pid;
            resp.action = 'G';
            int frep = open(next.reply, O_WRONLY);
            if (frep != -1) { write(frep, &resp, sizeof(resp)); close(frep); }
            else local_writer_active = 0; // rollback if failed
            printf("WC: granted writer %d from queue\n", next.pid);

            // Send sync updated state
            message sync;
            memset(&sync, 0, sizeof(sync));
            sync.action = 'S';
            sync.readers = remote_readers;
            sync.waiting_writers = (queue_end - queue_start + MAX_WAITING_WRITERS) % MAX_WAITING_WRITERS;
            sync.writer_active = local_writer_active;
            write(fifo_wc_to_rc_fd, &sync, sizeof(sync));
            printf("WC: sent sync -> writer_active=%d waiting_writers=%d readers=%d\n",
                   local_writer_active, sync.waiting_writers, remote_readers);
        }
    }

    close(fifo_ww_fd);
    close(fifo_rc_fd);
    close(fifo_wc_to_rc_fd);
    return 0;
}
