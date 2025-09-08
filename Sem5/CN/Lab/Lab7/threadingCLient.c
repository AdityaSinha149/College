#include "socketUtils.h"

char name[1024];

void listenAndPrintMessagesFromOtherClientsOnNewThread(int clientFD);

void *listenAndPrint(void *arg);

void sendMessageToServer(int clientFD);

void encryptMessage(char * buff);

void decryptMessage(char * buff, char * name);
int main() {
    int clientFD = makeIP4Socket();
    struct sockaddr_in servAddr = assignSocketAddress();
    if(connect(clientFD, (struct sockaddr *)&servAddr, sizeof(servAddr)) == -1) {
        perror("connect error : ");
        return 1;
    }
    
    printf("Who are you : ");
    fgets(name, sizeof(name), stdin);
    name[strcspn(name, "\n")] = '\0';

    send(clientFD, name, sizeof(name), 0);

    listenAndPrintMessagesFromOtherClientsOnNewThread(clientFD);

    sendMessageToServer(clientFD);

    return 0;
}

void listenAndPrintMessagesFromOtherClientsOnNewThread(int clientFD) {
    pthread_t id;
    int *fd = (int *)malloc(sizeof(int));
    *fd = clientFD;
    pthread_create(&id, NULL, listenAndPrint, fd);
    pthread_detach(id);
}

void *listenAndPrint(void *arg) {
    struct msg currMsg;
    char senderName[1024];
    int clientFD = *((int*)arg);
    free(arg);

    while(1) {
        ssize_t n = recv(clientFD, &currMsg, sizeof(currMsg), 0);

        decryptMessage(currMsg.msg, currMsg.name);
        if(n > 0) {
            printf("%s\n", currMsg.msg);
        } else if(n == 0) {
            printf("Server disconnected.\n");
            close(clientFD);
            break;
        }
    }
    return NULL;
}

void decryptMessage(char * buff, char * senderName) {
    int n = strlen(senderName);
    int m = strlen(buff);
    for (int i = 0; i < m; i++) {
        buff[i] -= senderName[i % n];
    }
}

void sendMessageToServer(int client) {
    char buff[1024];
    while(fgets(buff, sizeof(buff), stdin)) {
        buff[strcspn(buff, "\n")] = '\0';
        struct msg currMsg;
        strcpy(currMsg.name, name);
        sprintf(currMsg.msg, "%s : %s", name, buff);
        encryptMessage(currMsg.msg);
        send(client, &currMsg, sizeof(currMsg), 0);
        if(strncmp(buff, "bye", 3) == 0) exit(1);
    }
}

void encryptMessage(char * buff) {
    int n = strlen(name);
    int m = strlen(buff);
    for (int i = 0; i < m; i++) {
        buff[i] += name[i % n];
    }
}