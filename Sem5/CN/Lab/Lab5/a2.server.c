#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>
#include <netdb.h>

#define PORT 10200
#define IP "10.52.10.63"
#define BUFFER_SIZE 8192

int main() {
    int server_fd, client_fd;
    struct sockaddr_in serv_addr, cli_addr;
    socklen_t cli_addr_len = sizeof(cli_addr);
    char buffer[BUFFER_SIZE];
    char host[256], path[256];

    // Create TCP socket
    server_fd = socket(AF_INET, SOCK_STREAM, 0);
    if(server_fd < 0){ perror("socket failed"); exit(1); }

    serv_addr.sin_family = AF_INET;
    serv_addr.sin_port = htons(PORT);
    if(inet_pton(AF_INET, IP, &serv_addr.sin_addr) <= 0){
        perror("Invalid address / Address not supported");
        exit(1);
    }

    if(bind(server_fd, (struct sockaddr*)&serv_addr, sizeof(serv_addr)) < 0){
        perror("bind failed");
        close(server_fd);
        exit(1);
    }

    if(listen(server_fd, 3) < 0){
        perror("listen failed");
        close(server_fd);
        exit(1);
    }

    printf("Proxy Server listening on %s:%d...\n", IP, PORT);

    // Accept one client
    client_fd = accept(server_fd, (struct sockaddr*)&cli_addr, &cli_addr_len);
    if(client_fd < 0){ perror("accept failed"); exit(1); }

    // Ask client for URL and path
    char ask1[] = "Enter hostname (e.g., example.com): ";
    send(client_fd, ask1, strlen(ask1), 0);
    int n = read(client_fd, host, sizeof(host)-1);
    host[n] = '\0';

    char ask2[] = "Enter path (e.g., /, /index.html): ";
    send(client_fd, ask2, strlen(ask2), 0);
    n = read(client_fd, path, sizeof(path)-1);
    path[n] = '\0';

    printf("Client requested -> Host: %s, Path: %s\n", host, path);

    // ===== Act as client to real web server =====
    struct hostent *real_server = gethostbyname(host);
    if(real_server == NULL){
        char err[] = "ERROR: Could not resolve hostname\n";
        send(client_fd, err, strlen(err), 0);
        close(client_fd); close(server_fd);
        exit(1);
    }

    int proxy_sock = socket(AF_INET, SOCK_STREAM, 0);
    if(proxy_sock < 0){ perror("proxy socket failed"); exit(1); }

    struct sockaddr_in real_addr;
    memset(&real_addr, 0, sizeof(real_addr));
    real_addr.sin_family = AF_INET;
    real_addr.sin_port = htons(80);
    memcpy(&real_addr.sin_addr.s_addr, real_server->h_addr_list[0], real_server->h_length);

    if(connect(proxy_sock, (struct sockaddr*)&real_addr, sizeof(real_addr)) < 0){
        perror("connection to real server failed");
        exit(1);
    }

    // Send actual HTTP GET request to real server
    char request[1024];
    snprintf(request, sizeof(request),
             "GET %s HTTP/1.1\r\n"
             "Host: %s\r\n"
             "Connection: close\r\n\r\n", path, host);

    send(proxy_sock, request, strlen(request), 0);

    // Forward response to our client
    while((n = read(proxy_sock, buffer, sizeof(buffer))) > 0){
        send(client_fd, buffer, n, 0);
    }

    close(proxy_sock);
    close(client_fd);
    close(server_fd);
    return 0;
}
