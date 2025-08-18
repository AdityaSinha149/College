#include<stdio.h>
#include <stdlib.h>
#include<string.h>
#include <stdbool.h>
#include<sys/types.h>
#include<sys/socket.h>
#include<netinet/in.h>

#define PORT 10200
#define IP "172.16.48.109"

void remove_duplicates(char *buffer) {
    if (buffer == NULL) return;

    char seen[100][100];
    int seen_count = 0;
    char *token = strtok(buffer, " \t\n");
    char temp[1000] = "";
    
    while (token != NULL) {
        bool duplicate = false;
        for (int i = 0; i < seen_count; i++) {
            if (strcmp(seen[i], token) == 0) {
                duplicate = true;
                break;
            }
        }
        if (!duplicate) {
            strcat(temp, token);
            strcat(temp, " ");
            strcpy(seen[seen_count++], token);
        }
        token = strtok(NULL, " \t\n");
    }
    if (strlen(temp) > 0 && temp[strlen(temp)-1] == ' ')
        temp[strlen(temp)-1] = '\0';
    strcpy(buffer, temp);
}


int main(){
    int server_fd, new_socket;
    struct sockaddr_in address;
    int addrlen = sizeof(address);
    char buffer[1024] = {0};

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(server_fd == -1){
        perror("socket failed");
        exit(1);
    }

    address.sin_addr.s_addr = inet_addr(IP);
    address.sin_family = AF_INET;
    address.sin_port = htons(PORT);

    if(bind(server_fd,(struct sockaddr*)&address, sizeof(address)) < 0){
        perror("bind failed");
        exit(1);
    }

    if(listen(server_fd, 3) < 0){
        perror("listen failed");
        close(server_fd);
        exit(1);
    }

    printf("Server is listening on port : %d ip : %s socket %d", PORT, IP, server_fd);

    new_socket = accept(server_fd, (struct sockaddr*)&address, (socklen_t*)&addrlen);
    if(new_socket < 0){
        perror("accept failed");
        close(server_fd);
        exit(1);
    }

    read(new_socket, buffer, sizeof(buffer));
    if(strcmp(buffer, "STOP") == 0){
        close(new_socket);
        close(server_fd);
    }

    remove_duplicates(buffer);

    send(new_socket, buffer, sizeof(buffer), 0);


}