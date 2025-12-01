#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>
#include <time.h>

#define FIFO "/tmp/shelf_fifo"

int main() {
    // Create FIFO if it doesn't exist
    if (mkfifo(FIFO, 0666) == -1 && errno != EEXIST) {
        perror("mkfifo");
        exit(1);
    }

    int fd = open(FIFO, O_RDWR);
    if (fd == -1) {
        perror("open");
        exit(1);
    }

    int count = 5; // Start with 5 items
    printf("[Producer] Initial shelf count: %d\n", count);

    // Write initial count to FIFO
    if (write(fd, &count, sizeof(count)) == -1) {
        perror("write");
        close(fd);
        exit(1);
    }

    srand(time(NULL));

    while (1) {
        sleep(rand() % 5 + 1); // Wait 1-5 seconds

        // Read current count
        if (read(fd, &count, sizeof(count)) <= 0) {
            perror("read");
            break;
        }

        if (count < 5) {
            count++;
            printf("[Producer] Added item, shelf count: %d\n", count);
        } else {
            printf("[Producer] Shelf full (%d), waiting...\n", count);
        }

        // Write updated count
        if (write(fd, &count, sizeof(count)) == -1) {
            perror("write");
            break;
        }
    }

    close(fd);
    return 0;
}
