#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 10200
#define IP "10.55.4.245"

struct client_info {
    int fd;         // socket fd
    int key[16];    // encryption key
    int key_len;    // key length
};

// simple encrypt/decrypt functions
void encrypt(char *msg, int *key, int key_len) {
    int i, j = 0; key_len += 1;
    for (i = 0; msg[i] != '\0'; i++) {
        msg[i] = (msg[i] + key[j]) % 256;
        j = (j + 1) % key_len;
    }
}

void decrypt(char *msg, int *key, int key_len) {
    int i, j = 0; key_len += 1;
    for (i = 0; msg[i] != '\0'; i++) {
        msg[i] = (msg[i] - key[j] + 256) % 256;
        j = (j + 1) % key_len;
    }
}

int main() {
    struct client_info client;
    struct sockaddr_in server_addr, local_addr;
    socklen_t server_len = sizeof(server_addr), local_len = sizeof(local_addr);

    client.fd = socket(AF_INET, SOCK_STREAM, 0);
    if (client.fd < 0) { perror("socket"); exit(1); }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0) {
        perror("invalid ip"); exit(1);
    }

    if (connect(client.fd, (struct sockaddr *)&server_addr, server_len) < 0) {
        perror("connect"); close(client.fd); exit(1);
    }

    // get local IP and port
    getsockname(client.fd, (struct sockaddr *)&local_addr, &local_len);
    char ip[16]; inet_ntop(AF_INET, &local_addr.sin_addr, ip, sizeof(ip));
    int port = ntohs(local_addr.sin_port);

    // build key from IP and port
    client.key_len = 0;
    char *token = strtok(ip, ".");
    while (token) { client.key[client.key_len++] = atoi(token); token = strtok(NULL, "."); }

    int temp_port = port;
    while (temp_port) { client.key[client.key_len++] = temp_port % 100; temp_port /= 100; }

    // send key length
    int net_key_len = htonl(client.key_len);
    send(client.fd, &net_key_len, sizeof(int), 0);

    // send key
    for (int i = 0; i < client.key_len; i++) {
        int net_k = htonl(client.key[i]);
        send(client.fd, &net_k, sizeof(int), 0);
    }

    pid_t pid = fork();
    if (pid == 0) {
        // child: send messages to server
        char buff[1024];
        while (fgets(buff, sizeof(buff), stdin)) {
            encrypt(buff, client.key, client.key_len);
            send(client.fd, buff, strlen(buff), 0);
            if (strncmp(buff, "bye", 3) == 0) break;
        }
        exit(0);
    } else {
        // parent: receive messages from server
        char buff[1024];
        while (1) {
            int net_msg_len, n;
            n = recv(client.fd, &net_msg_len, sizeof(int), 0);
            if (n <= 0) break;
            int msg_len = ntohl(net_msg_len);

            int total = 0;
            while (total < msg_len) {
                n = recv(client.fd, buff + total, msg_len - total, 0);
                if (n <= 0) break;
                total += n;
            }
            buff[total] = '\0';

            // receive key length
            int net_key_len2;
            recv(client.fd, &net_key_len2, sizeof(int), 0);
            int dkey_len = ntohl(net_key_len2);

            // receive key
            int dkey[16];
            for (int i = 0; i < dkey_len; i++) {
                int net_k; recv(client.fd, &net_k, sizeof(int), 0);
                dkey[i] = ntohl(net_k);
            }

            decrypt(buff, dkey, dkey_len);
            printf("> %s", buff);
        }
    }

    close(client.fd);
}
