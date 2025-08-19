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
            printf("PID Child : %d\n", getpid());
            printf("PID Parent : %d\n", getppid());
            break;
        default :
            wait(&status);
            printf("Parent running\n");
            printf("PID Child : %d\n", pid);
            printf("PID Parent : %d\n", getpid());
            printf("Parent running : Child completed with status %d\n", status);
    }
    return 0;
}