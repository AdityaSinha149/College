#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/stat.h>
#include <string.h>
#include <errno.h>

#define FIFO "/tmp/my_fifo"

int main() {
    if (mkfifo(FIFO, 0666) == -1) {
        if (errno != EEXIST) {
            perror("mkfifo");
            exit(1);
        }
    }
    int fd = open(FIFO, O_RDONLY);
    if(fd == -1) {
        perror("open error");
        exit(1);
    }

    int a[4];
    printf("The numbers received are :\n");
    for(int i = 0; i < 4;i++) {
        if(read(fd, a + i, sizeof(int)) == -1) {
            perror("read error");
            close(fd);
            exit(1);
        }

        printf("%d ",a[i]);
    }
    printf("\n");

    close(fd);
    return 0;
}