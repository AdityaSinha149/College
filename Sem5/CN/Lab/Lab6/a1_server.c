#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <unistd.h>
#include <time.h>

#define IP "10.86.0.245"
#define PORT 10200

void connectChildToClient(int serverFD, struct sockaddr_in cliAddr, socklen_t cliAddrLen);

void sendDataToClient(int clientFD);

int main() {
    int serverFD = socket(AF_INET, SOCK_STREAM, 0);
    if(serverFD == -1) {
        perror("socket error");
        exit(1);
    }
    struct sockaddr_in servAddr, cliAddr;
    socklen_t cliAddrLen = sizeof(cliAddr);

    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, IP, &servAddr.sin_addr.s_addr);

    if(bind(serverFD, (struct sockaddr *)&servAddr, sizeof(servAddr)) == -1) {
        perror("bind error");
        return 1;
    }

    if(listen(serverFD, 10) == -1) {
        perror("listen error");
        return 1;
    }

    connectChildToClient(serverFD, cliAddr, cliAddrLen);

}

void connectChildToClient(int serverFD, struct sockaddr_in cliAddr, socklen_t cliAddrLen) {
    while(1) {
        int clientFD = accept(serverFD, (struct sockaddr *)&cliAddr, &cliAddrLen);
        if(clientFD == -1) {
            perror("accept error");
            exit(1);
        }

        if(fork() == 0) {
            //child
            sendDataToClient(clientFD);
        }
    }
}

void sendDataToClient(int clientFD) {
    time_t now = time(NULL);
    char *dt = ctime(&now);

    pid_t pid = getpid();

    char buffer[1024];
    snprintf(buffer, sizeof(buffer), "DateTime: %sServer PID: %d\n", dt, pid);

    send(clientFD, buffer, strlen(buffer), 0);
}