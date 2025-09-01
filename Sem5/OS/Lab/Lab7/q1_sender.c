#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

//make whatever type of struct u need/can
typedef struct buff {
    long int type;  //neccesary
    char txt[1024];
    int num;
} buff;

int main() {
    buff msg;
    int msgid = msgget((key_t)0666, 0666 | IPC_CREAT);
    if (msgid == -1) {
        perror("msgget error");
        exit(1);
    }

    printf("Enter the number to check Palindrome:\n");
    scanf("%d", &msg.num);
    msg.type = 1;       //msg type to segregate things

    if (msgsnd(msgid, (void *)&msg, sizeof(msg), 0) == -1) {
        perror("msgsnd error");
        exit(1);
    }

    if (msgrcv(msgid, (void *)&msg, sizeof(msg), 0, 0) == -1) {
        perror("msgrcv error");
        exit(1);
    }

    printf("It is %s\n", msg.txt);

    return 0;
}
