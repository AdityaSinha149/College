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
            printf("Running Q1 in child\n");
            execlp("./q1", "q1", (char *)0);
            printf("Child Completed\n");
            break;
        default :
            wait(&status);
            printf("Parent's Parent running : Child completed with status %d\n", status);
    }
    return 0;
}