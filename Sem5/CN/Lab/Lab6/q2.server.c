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
#define IP "10.203.228.195"

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
    int server,client;
    struct sockaddr_in server_addr, client_addr;
    socklen_t server_len = sizeof(server_addr);
    socklen_t client_len = sizeof(client_addr);

    //make socket
    server = socket(AF_INET, SOCK_STREAM, 0);
    if(server < 0) {
        perror("server error");
        exit(1);
    }

    //set server addr
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if(inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0) {
        perror("invalid ip address");
        close(server);
        exit(1);
    }

    //bind the server with socket
    if(bind(server, (struct sockaddr*)&server_addr, server_len)) {
        perror("bind error");
        close(server);
        exit(1);
    }
    
    printf("Server listening at %s:%d\n", IP, PORT);

    //listen to clients
    if(listen(server, 5) < 0) {
        perror("listen error");
        close(server);
        exit(1);
    }

    while(1) {
        //accept any connections
        client = accept(server, (struct sockaddr*)&client_addr, &client_len);
        if(client < 0) {
            perror("accept error");
            close(server);
            exit(1);
        }
        
        //child process
        if(fork() == 0){
            char buffer[1024];
            while(1){
                int valread = read(client, buffer, sizeof(buffer)-1);
                if(valread > 0){
                    buffer[valread] = '\0';
                    remove_duplicates(buffer);
                    send(client, buffer, strlen(buffer), 0);
                }
            }
        }
        else{
            close(client);
        }
    }

    close(server);
    return 0;
}