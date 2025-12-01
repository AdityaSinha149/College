#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 10200
#define IP "10.203.228.195"

int main(){
    int client;
    struct sockaddr_in server_addr;
    socklen_t server_len = sizeof(server_addr);
    
    //create socket
    client = socket(AF_INET, SOCK_STREAM, 0);
    if(client < 0) {
        perror("socket error");
        exit(1);
    }

    //setup server addr
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if(inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0) {
        perror("invalid ip address");
        exit(1);
    }

    // Connect to server
    if (connect(client, (struct sockaddr *)&server_addr, server_len) < 0) {
        perror("Connection failed");
        close(client);
        exit(1);
    }

    char msg[1024];
    char buffer[1024];
    while(1){
        printf("Enter message for server: ");
        fgets(msg, sizeof(msg), stdin);
        msg[strcspn(msg, "\r\n")] = 0; // remove newline and carriage return

        send(client, msg, strlen(msg), 0);

        memset(buffer, 0, sizeof(buffer));
        int valread = read(client, buffer, sizeof(buffer)-1);
        if (valread > 0) {
            buffer[valread] = '\0';
            printf("Server: %s\n", buffer);
        }
    }
    close(client);
    return 0;
}