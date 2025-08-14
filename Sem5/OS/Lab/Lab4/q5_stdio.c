#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <errno.h>

int main(int argc, char *argv[]) {
    if(argc!=3) {
        printf("Format: %s <Source_File> <Destination_File>\n", argv[0]);
        return 1;
    }

    FILE *src = fopen(argv[1], "r");
    if(src == NULL) {
        fprintf(stderr, "Cannot open %s: %s\n", argv[1], strerror(errno));
        return 1;
    }
    FILE *dst = fopen(argv[2], "w");
    if(dst == NULL) {
        fprintf(stderr, "Cannot open %s: %s\n", argv[2], strerror(errno));
        fclose(src);
        return 1;
    }
    

    int ch;
    while((ch = fgetc(src)) != EOF) {
        fputc(ch, dst);
    }

    fclose(src);
    fclose(dst);
    return 0;
}