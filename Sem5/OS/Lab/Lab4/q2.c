#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if (argc != 3) {
        write(2, "Usage: ./grep <word> <filename>\n", 33);
        exit(1);
    }

    char *word = argv[1];
    int fd = open(argv[2], O_RDONLY);
    if (fd < 0) {
        write(2, "Error: Cannot open file\n", 25);
        exit(1);
    }

    char buf[1024], line[2048];
    int n, i = 0;

    while ((n = read(fd, buf, sizeof(buf))) > 0) {
        for (int j = 0; j < n; j++) {
            if (buf[j] != '\n' && i < sizeof(line) - 1) {
                line[i++] = buf[j];
            } else {
                line[i] = '\0';
                if (strstr(line, word)) {
                    write(1, line, strlen(line));
                    write(1, "\n", 1);
                }
                i = 0;
            }
        }
    }

    if (i > 0) {
        line[i] = '\0';
        if (strstr(line, word)) {
            write(1, line, strlen(line));
            write(1, "\n", 1);
        }
    }

    close(fd);
    return 0;
}
