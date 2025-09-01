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

    int fd = open(FIFO, O_WRONLY);
    if(fd == -1) {
        perror("file open error");
        exit(1);
    }

    printf("Enter 4 numebrs : ");
    int a[4];
    scanf("%d %d %d %d",a, a+1, a+2, a+3);

    for(int i = 0; i < 4; i++) {
        if(write(fd, a + i, sizeof(int)) == -1) {
            perror("write error");
            close(fd);
            exit(1);
        }
    }

    printf("Numbers sent: %d, %d, %d, %d\n", a[0], a[1], a[2], a[3]);
    close(fd);
    return 0;
}