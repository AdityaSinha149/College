#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

void copy_part(int src, int dst, const char *choice) {
    int size = lseek(src, 0, SEEK_END);
    if(size == -1) {
        write(2, "Error calculating file size\n", 29);
        return;
    }

    int section = size / 4;
    int start_offset;

    if(strcmp(choice, "initial") == 0) {
        start_offset = 0;
    } 
    else if(strcmp(choice, "middle") == 0) {
        start_offset = section;
    } 
    else if(strcmp(choice, "last") == 0) {
        start_offset = size - section;
    } 
    else {
        write(1, "Invalid choice. Use initial/middle/last\n", 40);
        return;
    }

    lseek(src, start_offset, SEEK_SET);

    char byte;
    int remaining = section;
    while(remaining > 0 && read(src, &byte, 1) == 1) {
        write(dst, &byte, 1);
        remaining--;
    }
}

int main(int argc, char* argv[]) {
    if(argc < 4) {
        write(1, "Format: ./q5_copy_part <source> <destination> <choice>\n", 55);
        return 1;
    }

    int src = open(argv[1], O_RDONLY);
    if(src < 0) {
        write(1, "Source file not found\n", 22);
        return 1;
    }

    int dst = creat(argv[2], 0644);
    if(dst < 0) {
        write(1, "Error creating destination file\n", 32);
        close(src);
        return 1;
    }

    copy_part(src, dst, argv[3]);

    close(src);
    close(dst);

    return 0;
}
