#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <signal.h>

#define PORT 10200
#define IP "10.52.6.57"

void encrypt(char *msg, int *key, int key_len) {
    int i, j = 0;
    for (i = 0; msg[i] != '\0'; i++) {
        msg[i] = (msg[i] + key[j]) % 256;
        j = (j + 1) % key_len;
    }
}

void decrypt(char *msg, int *key, int key_len) {
    int i, j = 0;
    for (i = 0; msg[i] != '\0'; i++) {
        msg[i] = (msg[i] - key[j] + 256) % 256;
        j = (j + 1) % key_len;
    }
}

int main() {
    int server;
    struct sockaddr_in server_addr, client_addr;
    socklen_t server_len = sizeof(server_addr);
    socklen_t client_len = sizeof(client_addr);
    
    // create socket
    server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) {
        perror("socket error");
        exit(1);
    }

    // setup server addr
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0) {
        perror("invalid ip address");
        exit(1);
    }

    // connect to server
    if (connect(server, (struct sockaddr *)&server_addr, server_len) < 0) {
        perror("Connection failed");
        close(server);
        exit(1);
    }
    
    getsockname(server, (struct sockaddr*)&client_addr, &client_len);

    char ip[16];
    inet_ntop(AF_INET, &client_addr.sin_addr, ip, sizeof(ip));
    int port = ntohs(client_addr.sin_port);

    printf("%s:%d\n", ip,port);
    int key[16];
    int key_len = 0;

    //build key from IP
    char *token = strtok(ip, ".");
    while (token) {
        key[key_len++] = atoi(token);
        token = strtok(NULL, ".");
    }

    //add port to key
    while (port) {
        key[key_len++] = port % 100;
        port /= 100;
    }

    //change to n form
    int net_key_len = htonl(key_len);

    //send key
    if (send(server, &net_key_len, sizeof(int), 0) != sizeof(int)) {
    perror("send key length");
    exit(1);
    }

    // send exactly 16 ints (pre-zeroed)
    int net_key[16] = {0};
    for (int i = 0; i < key_len; i++) {
        net_key[i] = htonl(key[i]);
        send(server, &net_key[i], sizeof(int), 0);
    }

    pid_t pid = fork();
    if (pid == 0) {
        //client's child sends msgs to server's child
        while (1) {
            char buff[1024];
            if (!fgets(buff, sizeof(buff), stdin)) break;

            encrypt(buff, key, key_len); //encrypt
            send(server, buff, strlen(buff), 0);
            
            if (strncmp(buff, "bye", 3) == 0) break; //exit
        }
        kill(getppid(), SIGTERM);
        return 0;
    } else {
        //client's parent listen to other clients' children
        char buff[1024];
        while (1) {
            int n = recv(server, buff, sizeof(buff)-1, 0);
            if (n <= 0) break;
            buff[n] = '\0';

            //receive decryption key
            int dkey_len;
            int dkey[16];
            recv(server, &dkey_len, sizeof(int), 0);
            recv(server, dkey, 16 * sizeof(int), 0);
            dkey_len = ntohl(dkey_len);
            for(int i = 0; i < dkey_len; i++){
                dkey[i] = ntohl(dkey[i]);
                printf("%d ", dkey[i]);
            }
            decrypt(buff, dkey, dkey_len); //decrypt
            printf("> %s\n", buff);
        }
    }

    close(server);
}