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
#define MAX_CLIENTS 1024

struct client_info {
    int fd;             // socket fd
    int id;             // unique client id
    int key[16];        // encryption key
    int key_len;        // key length
};

int main() {
    int server;
    struct sockaddr_in server_addr, client_addr;
    socklen_t server_len = sizeof(server_addr);
    socklen_t client_len = sizeof(client_addr);

    struct client_info clients[MAX_CLIENTS];
    int cli_count = 0;

    server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0) { perror("socket error"); exit(1); }

    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0) {
        perror("invalid ip"); close(server); exit(1);
    }

    if (bind(server, (struct sockaddr *)&server_addr, server_len)) {
        perror("bind error"); close(server); exit(1);
    }

    printf("Server listening at %s:%d\n", IP, PORT);
    if (listen(server, 5) < 0) { perror("listen error"); close(server); exit(1); }

    while (1) {
        int client_fd = accept(server, (struct sockaddr *)&client_addr, &client_len);
        if (client_fd < 0) { perror("accept error"); continue; }

        // initialize client struct
        struct client_info *c = &clients[cli_count];
        c->fd = client_fd;
        c->id = cli_count;

        printf("Client %d connected (fd=%d)\n", c->id, c->fd);

        // receive key length
        int net_len;
        recv(c->fd, &net_len, sizeof(int), 0);
        c->key_len = ntohl(net_len);

        // receive key
        for (int i = 0; i < c->key_len; i++) {
            int net_k; recv(c->fd, &net_k, sizeof(int), 0);
            c->key[i] = ntohl(net_k);
        }

        int this_client = c->fd;
        int this_id = c->id;
        int this_key_len = c->key_len;
        int this_key[16];
        memcpy(this_key, c->key, sizeof(int) * 16);

        if (fork() == 0) {
            close(server); // child closes listening socket
            char buff[1024];

            while (1) {
                int n = read(this_client, buff, sizeof(buff) - 1);
                if (n <= 0) break;
                buff[n] = '\0';

                // broadcast to all other clients
                for (int i = 0; i < cli_count; i++) {
                    if (clients[i].fd <= 0 || clients[i].id == this_id) continue;

                    int net_msg_len = htonl(n);
                    send(clients[i].fd, &net_msg_len, sizeof(int), 0);
                    send(clients[i].fd, buff, n, 0);

                    int net_key_len = htonl(this_key_len);
                    send(clients[i].fd, &net_key_len, sizeof(int), 0);
                    for (int j = 0; j < this_key_len; j++) {
                        int net_k = htonl(this_key[j]);
                        send(clients[i].fd, &net_k, sizeof(int), 0);
                    }
                }

                if (strncmp(buff, "bye", 3) == 0) break;
            }

            close(this_client);
            exit(0);
        } else {
            cli_count++;
        }
    }
}
