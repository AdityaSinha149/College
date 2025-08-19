#include <sys/types.h>
#include <sys/wait.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int main(){
    int pid = fork();
    int status;
    switch(pid){
        case -1 :
            perror("fork failed");
            exit(1);
        case 0 :
            printf("Child running\n");
            break;
        default :
            wait(&status);
            printf("Parent running : Child completed with status %d\n", status);
    }
    return 0;
}