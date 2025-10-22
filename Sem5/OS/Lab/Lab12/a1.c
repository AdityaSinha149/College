#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>

int main() {
    DIR *dir;
    struct dirent *entry;
    struct stat st;
    char **dirs = NULL;
    int count = 0;

    // Open current directory
    dir = opendir(".");
    if (!dir) {
        perror("opendir");
        return 1;
    }

    // Read entries
    while ((entry = readdir(dir)) != NULL) {
        // Skip . and ..
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0)
            continue;

        // Get info about the entry
        if (stat(entry->d_name, &st) == 0) {
            if (S_ISDIR(st.st_mode)) { // Check if it is a directory
                dirs = realloc(dirs, sizeof(char*) * (count + 1));
                dirs[count] = strdup(entry->d_name);
                count++;
            }
        }
    }
    closedir(dir);

    // Sort directories alphabetically
    for (int i = 0; i < count - 1; i++) {
        for (int j = i + 1; j < count; j++) {
            if (strcmp(dirs[i], dirs[j]) > 0) {
                char *tmp = dirs[i];
                dirs[i] = dirs[j];
                dirs[j] = tmp;
            }
        }
    }

    // Print directories
    printf("Subdirectories in alphabetical order:\n");
    for (int i = 0; i < count; i++) {
        printf("%s\n", dirs[i]);
        free(dirs[i]); // Free strdup
    }
    free(dirs);

    return 0;
}
