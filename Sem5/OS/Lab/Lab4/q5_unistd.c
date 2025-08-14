#include <unistd.h>
#include <fcntl.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    if(argc!=3) {
        char msg[512];
        snprintf(msg, sizeof(msg), "Format: %s <Source_File> <Destination_File>\n", argv[0]);
        write(1, msg, sizeof(msg));
        return 1;
    }

    int src = open(argv[1], O_RDONLY);
    if(src < 0) {
        write(1, "Cannot open Source", 18);
        return 1;
    }
    int dst = open(argv[2], O_WRONLY);
    if(dst < 0) {
        write(1, "Cannot open Destination", 23);
        close(src);
        return 1;
    }
    

    char ch;
    while((read(src, &ch, 1)) == 1) {
        write(dst, &ch, 1);
    }

    close(src);
    close(dst);
    return 0;
}