#include "socketUtils.h"

char name[1024];

void listenAndPrintMessagesFromOtherClientsOnNewThread(int clientFD);
void *listenAndPrint(void *arg);
void sendMessageToServer(int clientFD);

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
    char buff[1024];
    int clientFD = *((int*)arg);
    free(arg);

    while(1) {
        ssize_t n = recv(clientFD, buff, 1024, 0);
        if(n > 0) {
            buff[n] = 0;
            printf("%s\n", buff);
        } else if(n == 0) {
            printf("Server disconnected.\n");
            close(clientFD);
            break;
        }
    }
    return NULL;
}

void sendMessageToServer(int client) {
    char buff[1024];
    while(fgets(buff, sizeof(buff), stdin)) {
        buff[strcspn(buff, "\n")] = '\0';
        char msg[1024];
        sprintf(msg, "%s : %s", name, buff);
        send(client, msg, strlen(msg), 0);
    }
}
