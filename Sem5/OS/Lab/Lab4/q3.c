#include <fcntl.h>
#include <unistd.h>
#include <string.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    if(argc < 2){
        write(1, "Format: ./more <file1> <file2> ...\n", 36);
        return 1;
    }

    for(int i = 1; i < argc; i++) {
        int fd = open(argv[i], O_RDONLY);
        if (fd < 0) {
            write(2, "Cannot open file\n", 18);
            continue;
        }

        write(1, "\n--- FILE: ", 11);
        write(1, argv[i], strlen(argv[i]));
        write(1, " ---\n", 6);

        char ch, line[1024];
        int idx = 0, count = 0;
        while(read(fd,&ch,1) == 1) {
            line[idx++] = ch;
            if(ch == '\n' || idx==1023) {
                line[idx] = '\0';
                write(1,line,strlen(line));
                idx=0;
                count++;

                if(count == 20) {
                    write(1, "\n--More-- Press Enter to continue--", 35);
                    char temp;
                    read(0, &temp, 1);
                    count=0;
                    write(1,"\n",1);
                }
            }
        }

        if(idx>0) {
            write(1, "\n--More-- Press Enter to continue--", 35);
            line[idx] = '\0';
            write(1,line,strlen(line));
        }
        
        close(fd);
    }
    return 0;
}