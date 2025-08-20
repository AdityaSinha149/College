#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <stdbool.h>

#define PORT 10200
#define IP "10.52.10.63"

void encrypt(char * msg){
    int i = 0;
    while(msg[i] != '\0')
        msg[i++] += 4;
}

int main(){
    int client;
    struct sockaddr_in server_addr;

    client = socket(AF_INET, SOCK_STREAM, 0);
    if(client < 0){
        perror("Socket creation failed");
        exit(1);
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);

    // Use inet_pton instead of inet_addr
    if(inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0){
        perror("Invalid address/ Address not supported");
        exit(1);
    }

    printf("Trying to connect to server at %s:%d\n", IP, PORT);
    if(connect(client, (struct sockaddr*)&server_addr, sizeof(server_addr)) < 0){
        perror("Connection Failed");
        close(client);
        exit(1);
    }

    printf("Enter Text: ");
    char msg[1024];
    fgets(msg, sizeof(msg), stdin);

    encrypt(msg);
    send(client, msg, strlen(msg), 0);

    close(client);
    return 0;
}
