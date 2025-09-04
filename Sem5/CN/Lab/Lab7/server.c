#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 10200
#define IP "10.52.6.57"

int main()
{
    int server, client[1024];
    struct sockaddr_in server_addr, client_addr;
    socklen_t server_len = sizeof(server_addr);
    socklen_t client_len = sizeof(client_addr);
    int cli_idx = 0;
    int key[1024][16];
    int key_len[1024];

    // make socket
    server = socket(AF_INET, SOCK_STREAM, 0);
    if (server < 0)
    {
        perror("server error");
        exit(1);
    }

    // set server addr
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if (inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0)
    {
        perror("invalid ip address");
        close(server);
        exit(1);
    }

    // bind the server with socket
    if (bind(server, (struct sockaddr *)&server_addr, server_len))
    {
        perror("bind error");
        close(server);
        exit(1);
    }

    printf("Server listening at %s:%d\n", IP, PORT);

    // listen to clients
    if (listen(server, 5) < 0)
    {
        perror("listen error");
        close(server);
        exit(1);
    }

    while (1)
    {
        client[cli_idx] = accept(server, (struct sockaddr *)&client_addr, &client_len);
        if (client[cli_idx] < 0)
        {
            perror("accept error");
            return 1;
        }

        recv(client[cli_idx], &key_len[cli_idx], sizeof(int), 0);
        int size = ntohl(key_len[cli_idx]);
        printf("size: %d" , size);

        for(int i = 0; i < size; i++){
            recv(client[cli_idx], &key[cli_idx][i], sizeof(int), 0);
        }
        printf("received");

        if (fork() == 0)
        {
            // child
            close(server);
            char buff[1024];
            int n;
            while ((n = read(client[cli_idx], buff, sizeof(buff) - 1)) > 0)
            {
                buff[n] = '\0';
                for (int i = 0; i < 1024; i++)
                {
                    if (client[i] <= 0)
                        continue;
                    send(client[i], buff, sizeof(buff), 0);
                    send(client[i], &key_len[cli_idx], sizeof(int), 0);
                    send(client[i], key[cli_idx], 16 * sizeof(int), 0);
                }
                if (strncmp(buff, "bye", 3) == 0) break;
            }
            close(client[cli_idx]);
        }

        else
        {
            close(client[cli_idx]);
            cli_idx++;
        }
    }
}