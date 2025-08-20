#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define PORT 10200
#define IP "10.52.10.63"
#define BUFFER_SIZE 8192

int main() {
    int client, n;
    struct sockaddr_in serv_addr;
    socklen_t serv_addr_len = sizeof(serv_addr);
    char buffer[BUFFER_SIZE];
    char input[256];

    // Create TCP socket
    client = socket(AF_INET, SOCK_STREAM, 0);
    if(client < 0){ perror("socket failed"); exit(1); }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    if(inet_pton(AF_INET, IP, &serv_addr.sin_addr) <= 0){
        perror("Invalid address / Address not supported");
        exit(1);
    }

    if(connect(client, (struct sockaddr*)&serv_addr, serv_addr_len) < 0){
        perror("connection failed");
        close(client);
        exit(1);
    }

    // First prompt: hostname
    n = read(client, buffer, sizeof(buffer)-1);
    buffer[n] = '\0';
    printf("%s", buffer);
    scanf("%255s", input);
    send(client, input, strlen(input), 0);

    // Second prompt: path
    n = read(client, buffer, sizeof(buffer)-1);
    buffer[n] = '\0';
    printf("%s", buffer);
    scanf("%255s", input);
    send(client, input, strlen(input), 0);

    // Receive HTTP response
    printf("\n--- HTTP Response from real server ---\n");
    int header_done = 0;
    while((n = read(client, buffer, sizeof(buffer)-1)) > 0){
        buffer[n] = '\0';
        if (!header_done) {
            char *sep = strstr(buffer, "\r\n\r\n");
            if (sep) {
                *sep = '\0';
                printf("\n[HTTP Headers]\n%s\n", buffer);
                printf("\n[HTTP Body]\n%s\n", sep + 4);
                header_done = 1;
            } else {
                printf("%s", buffer);
            }
        } else {
            printf("%s", buffer);
        }
    }

    close(client);
    return 0;
}
