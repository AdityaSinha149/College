#include "socketUtils.h"

struct client {
    int clientFD;
    struct sockaddr_in clientAddr;
    int key[16];
    char name[1024];
};

struct client clients[1024];
int clientCount = 0;

void acceptClients(int serverFD);

struct client acceptIncomingClient(int serverFD);

void receiveAndPrintIncomingDataOnNewThread(struct client currClient);

void *receiveAndPrintIncomingData(void *arg);

void sendToAllOtherClients(char *msg, int senderFD, char * senderName);

int main() {
    int serverFD = makeIP4Socket();
    struct sockaddr_in servAddr = assignSocketAddress();
    
    if(bind(serverFD, (struct sockaddr *)&servAddr, sizeof(servAddr)) == -1) {
        perror("bind error : ");
        return 1;
    }

    if(listen(serverFD, 10) == -1) {
        perror("listen error");
        return 1;
    }

    printf("Server is Listening at %s:%d\n", IP, PORT);

    acceptClients(serverFD);
    
    return 0;
}

void acceptClients(int serverFD) {
    while(1) {
        struct client currentClient = acceptIncomingClient(serverFD);
        clients[clientCount++] = currentClient;
        //start a new thread for listening and printing data
        receiveAndPrintIncomingDataOnNewThread(currentClient);
    }
}

struct client acceptIncomingClient(int serverFD) {
    struct sockaddr_in clientAddr;
    socklen_t clientAddrLen = sizeof(clientAddr);
    int clientFD = accept(serverFD, (struct sockaddr*)&clientAddr, &clientAddrLen);
    if(clientFD == -1) {
        perror("accept error : ");
        exit(1);
    }
    struct client currClient;
    currClient.clientAddr = clientAddr;
    currClient.clientFD = clientFD;
    recv(clientFD, currClient.name, sizeof(currClient.name), 0);
    return currClient;
}

void receiveAndPrintIncomingDataOnNewThread(struct client currClient) {
    pthread_t id;
    struct client *newClient = (struct client *)malloc(sizeof(struct client));
    *newClient = currClient;
    pthread_create(&id, NULL, receiveAndPrintIncomingData, newClient);
    pthread_detach(id);
}

void *receiveAndPrintIncomingData(void *arg) {
    struct msg currMsg;
    struct client currClient = *((struct client*)arg);
    free(arg); // free after copying
    int clientFD = currClient.clientFD;

    while(1) {
        ssize_t n = recv(clientFD, &currMsg, sizeof(currMsg), 0);
        
        if(n > 0) {
            printf("%s\n", currMsg.msg);
            sendToAllOtherClients(currMsg.msg, clientFD, currMsg.name);
        } else if(n == 0) {
            printf("Client disconnected.\n");
            close(clientFD);
            break;
        }
    }
    return NULL;
}

void sendToAllOtherClients(char *msg, int senderFD, char * senderName) {
    for(int i = 0; i < clientCount; i++) {
        if(clients[i].clientFD == -1) continue;

        if(strncmp(msg, "bye", 3) == 0 && clients[i].clientFD == senderFD) {
            close(clients[i].clientFD);
            clients[i].clientFD = -1;
        } 
        else if(clients[i].clientFD != senderFD) {
            struct msg currMsg;
            strcpy(currMsg.msg, msg);
            strcpy(currMsg.name, senderName);
            send(clients[i].clientFD, &currMsg, sizeof(currMsg), 0);
        }
    }
}
