#include<stdio.h>
#include<string.h>
#include<stdlib.h>
#include<unistd.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>
#include<arpa/inet.h> 

#define PORT 10200
#define IP "10.154.184.195"

int main() {
    int sock = 0;
    struct sockaddr_in serv_addr;
    char buffer[1024] = {0};
    char msg[1024];

    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket failed");
        exit(1);
    }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);

    if (inet_pton(AF_INET, IP, &serv_addr.sin_addr) <= 0) {
        perror("Invalid address/ Address not supported");
        exit(1);
    }

    if (connect(sock, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0) {
        perror("Connection Failed");
        exit(1);
    }

    printf("Enter message for server: ");
    scanf("%1023[^\n]", msg);

    send(sock, msg, strlen(msg), 0);
    printf("Message sent to server: %s\n", msg);

    int valread = read(sock, buffer, sizeof(buffer)-1);
    if (valread > 0) {
        buffer[valread] = '\0';
        printf("Server: %s\n", buffer);
    }

    close(sock);
    return 0;
}
