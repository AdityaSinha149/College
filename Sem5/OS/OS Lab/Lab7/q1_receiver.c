#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <errno.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>

typedef struct buff {
    long int type;
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

    if (msgrcv(msgid, (void *)&msg, sizeof(msg), 0, 0) == -1) {
        perror("msgrcv error");
        exit(1);
    }

    int n = msg.num, rev = 0, temp = n;
    while (temp > 0) {
        rev = rev * 10 + (temp % 10);
        temp /= 10;
    }

    if (n == rev)
        strcpy(msg.txt, "a palindrome");
    else
        strcpy(msg.txt, "not a palindrome");

    msg.type = 1;

    if (msgsnd(msgid, (void *)&msg, sizeof(msg), 0) == -1) {
        perror("msgsnd error");
        exit(1);
    }

    printf("Processed number: %d\n", msg.num);

    return 0;
}
