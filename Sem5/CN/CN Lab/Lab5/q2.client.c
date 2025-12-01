#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define PORT 10200
#define SERVER_IP "10.154.184.195"
#define MAX_BUFFER 1024

int main() {
    int sockfd;
    struct sockaddr_in servaddr;
    socklen_t len = sizeof(servaddr);
    char buffer[MAX_BUFFER];
    int rows, cols;

    // Create UDP socket
    sockfd = socket(AF_INET, SOCK_DGRAM, 0);
    if(sockfd < 0) { perror("socket failed"); exit(1); }

    servaddr.sin_family = AF_INET;
    servaddr.sin_port = htons(PORT);
    servaddr.sin_addr.s_addr = inet_addr(SERVER_IP);

    // Initial handshake
    char init_msg[] = "HELLO";
    sendto(sockfd, init_msg, strlen(init_msg)+1, 0, (struct sockaddr*)&servaddr, len);

    // Receive prompt for rows
    recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr*)&servaddr, &len);
    printf("%s\n> ", buffer);
    scanf("%d", &rows);
    int rows_network = htonl(rows);
    sendto(sockfd, &rows_network, sizeof(rows_network), 0, (struct sockaddr*)&servaddr, len);

    // Receive prompt for columns
    recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr*)&servaddr, &len);
    printf("%s\n> ", buffer);
    scanf("%d", &cols);
    int cols_network = htonl(cols);
    sendto(sockfd, &cols_network, sizeof(cols_network), 0, (struct sockaddr*)&servaddr, len);

    // Send each row
    for(int i = 0; i < rows; i++) {
        recvfrom(sockfd, buffer, sizeof(buffer), 0, (struct sockaddr*)&servaddr, &len);
        printf("%s\n> ", buffer);

        getchar(); // consume leftover newline
        char row_input[MAX_BUFFER];
        fgets(row_input, sizeof(row_input), stdin);
        row_input[strcspn(row_input, "\n")] = 0;

        sendto(sockfd, row_input, strlen(row_input)+1, 0, (struct sockaddr*)&servaddr, len);
    }

    close(sockfd);
    return 0;
}
