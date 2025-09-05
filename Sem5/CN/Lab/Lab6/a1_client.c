#include <stdio.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <arpa/inet.h>
#include <string.h>
#include <stdlib.h>
#include <netinet/in.h>
#include <unistd.h>

#define IP "10.86.0.245"
#define PORT 10200

int main() {
    int clientFD = socket(AF_INET, SOCK_STREAM, 0);
    if(clientFD == -1) {
        perror("socket error");
        exit(1);
    }
    struct sockaddr_in servAddr;

    servAddr.sin_family = AF_INET;
    servAddr.sin_port = htons(PORT);
    inet_pton(AF_INET, IP, &servAddr.sin_addr.s_addr);

    if(connect(clientFD, (struct sockaddr *)&servAddr, sizeof(servAddr)) == -1) {
        perror("connect error");
        exit(1);
    }

    char buff[1024];
    int n = recv(clientFD, buff, sizeof(buff), 0);
    buff[n] = 0;

    printf("%s", buff);
    sleep(20);

    return 0;
}