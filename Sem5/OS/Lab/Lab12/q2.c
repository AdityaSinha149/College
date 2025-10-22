#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <sys/stat.h>
#include <string.h>

void listFiles(const char *path) {
    struct dirent *entry;
    DIR *dp = opendir(path);

    if (dp == NULL) {
        perror("opendir");
        return;
    }

    while ((entry = readdir(dp)) != NULL) {
        // Skip "." and ".."
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;

        char fullPath[1024];
        snprintf(fullPath, sizeof(fullPath), "%s/%s", path, entry->d_name);

        struct stat statbuf;
        if (stat(fullPath, &statbuf) == -1) {
            perror("stat");
            continue;
        }

        if (S_ISDIR(statbuf.st_mode)) {
            // It's a directory, recurse
            listFiles(fullPath);
        } else {
            // It's a file, print it
            printf("%s\n", fullPath);
        }
    }

    closedir(dp);
}

int main() {
    listFiles(".");
    return 0;
}

// ./a1.c
// ./q1.c
// ./q2.c
// ./q3.png
// ./q3.sh
// ./q4.png
// ./q4.sh
// ./a1
// ./q1
// ./q2