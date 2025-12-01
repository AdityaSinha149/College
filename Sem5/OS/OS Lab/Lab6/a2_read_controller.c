#include "a2_common.h"
#include <fcntl.h>
#include <unistd.h>
#include <stdio.h>
#include <sys/select.h>
#include <string.h>
#include <errno.h>

int main() {
    ensure_fifo("fifo_rr");         // Reader requests
    ensure_fifo("fifo_rc_to_wc");   // RC -> WC
    ensure_fifo("fifo_wc_to_rc");   // WC -> RC

    int local_readers = 0;           // readers controlled by RC locally
    int remote_writer_active = 0;    // state from WC
    int remote_waiting_writers = 0;  // state from WC

    printf("Reader Controller started (writer-priority)\n");

    // Persistent FIFO descriptors
    int fifo_rr_fd = open("fifo_rr", O_RDONLY | O_NONBLOCK);
    int fifo_wc_fd = open("fifo_wc_to_rc", O_RDWR | O_NONBLOCK); // O_RDWR prevents ENXIO
    int fifo_rc_to_wc_fd = open("fifo_rc_to_wc", O_RDWR | O_NONBLOCK);
    if (fifo_rr_fd == -1 || fifo_wc_fd == -1 || fifo_rc_to_wc_fd == -1) { perror("open fifo"); return 1; }

    while (1) {
        fd_set rfds;
        FD_ZERO(&rfds);
        FD_SET(fifo_rr_fd, &rfds);
        FD_SET(fifo_wc_fd, &rfds);
        int maxfd = (fifo_rr_fd > fifo_wc_fd ? fifo_rr_fd : fifo_wc_fd) + 1;

        if (select(maxfd, &rfds, NULL, NULL, NULL) <= 0) continue;

        // 1) Process all WC sync messages
        if (FD_ISSET(fifo_wc_fd, &rfds)) {
            message sync;
            while (read(fifo_wc_fd, &sync, sizeof(sync)) == sizeof(sync)) {
                if (sync.action == 'S') {
                    remote_writer_active = sync.writer_active;
                    remote_waiting_writers = sync.waiting_writers;
                    printf("RC: received sync -> writer_active=%d waiting_writers=%d local_readers=%d\n",
                           remote_writer_active, remote_waiting_writers, local_readers);
                }
            }
        }

        // 2) Process all reader requests
        if (FD_ISSET(fifo_rr_fd, &rfds)) {
            message req;
            while (read(fifo_rr_fd, &req, sizeof(req)) == sizeof(req)) {
                if (req.action == 'R') {
                    message resp;
                    memset(&resp, 0, sizeof(resp));

                    if (remote_writer_active || remote_waiting_writers > 0) {
                        // Deny reader
                        resp.pid = req.pid;
                        resp.action = 'D';
                        int frep = open(req.reply, O_WRONLY);
                        if (frep != -1) { write(frep, &resp, sizeof(resp)); close(frep); }
                        printf("RC: denied reader %d (writer active/waiting)\n", req.pid);
                    } else {
                        // Grant reader
                        local_readers++;
                        resp.pid = req.pid;
                        resp.action = 'G';
                        int frep = open(req.reply, O_WRONLY);
                        if (frep != -1) { write(frep, &resp, sizeof(resp)); close(frep); }
                        else local_readers--; // rollback if reply fails
                        printf("RC: granted reader %d (local_readers=%d)\n", req.pid, local_readers);

                        // Send sync to WC
                        message sync;
                        memset(&sync, 0, sizeof(sync));
                        sync.action = 'S';
                        sync.readers = local_readers;
                        sync.waiting_writers = 0; // RC doesn't track waiting_writers locally
                        sync.writer_active = 0;   // RC doesn't track writer_active locally
                        write(fifo_rc_to_wc_fd, &sync, sizeof(sync));
                        printf("RC: sent sync -> readers=%d\n", local_readers);
                    }
                } else if (req.action == 'r') {
                    if (local_readers > 0) local_readers--;
                    printf("RC: reader %d released (local_readers=%d)\n", req.pid, local_readers);

                    // Send sync to WC
                    message sync;
                    memset(&sync, 0, sizeof(sync));
                    sync.action = 'S';
                    sync.readers = local_readers;
                    sync.waiting_writers = 0;
                    sync.writer_active = 0;
                    write(fifo_rc_to_wc_fd, &sync, sizeof(sync));
                    printf("RC: sent sync -> readers=%d\n", local_readers);
                }
            }
        }
    }

    close(fifo_rr_fd);
    close(fifo_wc_fd);
    close(fifo_rc_to_wc_fd);
    return 0;
}
