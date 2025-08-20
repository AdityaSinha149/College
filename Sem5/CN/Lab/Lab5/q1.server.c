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

int main() {
    int server_fd, new_socket;
    struct sockaddr_in address;
    socklen_t addrlen = sizeof(address);
    char buffer[1024] = {0};

    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(server_fd == -1){
        perror("socket failed");
        exit(1);
    }

    address.sin_family = AF_INET;
    address.sin_port = htons(PORT);

    // Use inet_pton instead of inet_addr
    if(inet_pton(AF_INET, IP, &address.sin_addr) <= 0){
        perror("Invalid address / Address not supported");
        exit(1);
    }

    if(bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0){
        perror("bind failed");
        exit(1);
    }

    printf("Server is listening on %s:%d\n", IP, PORT);
    fflush(stdout);  // ensure message prints immediately

    if(listen(server_fd, 3) < 0){
        perror("listen failed");
        close(server_fd);
        exit(1);
    }
    
    new_socket = accept(server_fd, (struct sockaddr*)&address, &addrlen);
    if(new_socket < 0){
        perror("accept failed");
        close(server_fd);
        exit(1);
    }

    while(1){
        int valread = read(new_socket, buffer, sizeof(buffer)-1);
        if(valread > 0){
            buffer[valread] = '\0';
            if(strcmp(buffer, "STOP") == 0){
                close(new_socket);
                close(server_fd);
                break;
            }
            remove_duplicates(buffer);
            send(new_socket, buffer, strlen(buffer), 0);
        }
    }

    return 0;
}
