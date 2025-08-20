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

void decrypt(char * msg){
    int i = 0;
    while(msg[i] != '\0')
        msg[i++] -= 4;
}

int main(){
    int server, client;
    struct sockaddr_in addr;
    socklen_t addrlen = sizeof(addr);

    server = socket(AF_INET, SOCK_STREAM, 0);
    if(server < 0){
        perror("Socket creation failed");
        exit(1);
    }

    addr.sin_family = AF_INET;
    addr.sin_port = htons(PORT);

    // Use inet_pton instead of inet_addr as it is deprecated
    if(inet_pton(AF_INET, IP, &addr.sin_addr) <= 0){
        perror("Invalid address / Address not supported");
        exit(1);
    }

    if(bind(server, (struct sockaddr*)&addr, addrlen) < 0){
        perror("Bind failed");
        exit(1);
    }

    printf("Server is listening on %s:%d\n", IP, PORT);

    if(listen(server, 3) < 0){
        perror("Listen failed");
        close(server);
        exit(1);
    }

    client = accept(server, (struct sockaddr*)&addr, &addrlen);
    if(client < 0){
        perror("Accept failed");
        close(server);
        exit(1);
    }

    char buff[1024];
    int bytes_read = read(client, buff, sizeof(buff) - 1); // leave space for null terminator
    if(bytes_read < 0){
        perror("Read error");
        close(client);
        close(server);
        exit(1);
    }

    buff[bytes_read] = '\0';
    printf("Encrypted Message: %s\n", buff);

    decrypt(buff);
    printf("Decrypted Message: %s\n", buff);

    close(client);
    close(server);
    return 0;
}
