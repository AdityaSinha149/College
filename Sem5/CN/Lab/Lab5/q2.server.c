#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <arpa/inet.h>
#include <sys/socket.h>

#define PORT 10200
#define MAX_BUFFER 1024
#define MAX_MATRIX 100
#define IP "10.154.184.195"

int main() {
    int server_fd;
    struct sockaddr_in address, cli_address;
    socklen_t len = sizeof(cli_address);
    char buffer[MAX_BUFFER];
    int rows, cols;
    int matrix[MAX_MATRIX][MAX_MATRIX];

    // Create UDP socket
    server_fd = socket(AF_INET, SOCK_DGRAM, 0);
    if(server_fd < 0) {
        perror("socket failed");
        exit(1);
    }

    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr(IP);
    address.sin_port = htons(PORT);

    // Bind
    if(bind(server_fd, (struct sockaddr*)&address, sizeof(address)) < 0) {
        perror("bind failed");
        close(server_fd);
        exit(1);
    }

    printf("Server listening on port %d...\n", PORT);

    // Wait for initial message from client
    int n = recvfrom(server_fd, buffer, sizeof(buffer), 0, (struct sockaddr*)&cli_address, &len);
    if(n < 0) { perror("recvfrom"); exit(1); }

    // Receive number of rows
    strcpy(buffer, "Enter number of rows:");
    sendto(server_fd, buffer, strlen(buffer)+1, 0, (struct sockaddr*)&cli_address, len);

    recvfrom(server_fd, &rows, sizeof(rows), 0, (struct sockaddr*)&cli_address, &len);
    rows = ntohl(rows);

    // Receive number of columns
    strcpy(buffer, "Enter number of columns:");
    sendto(server_fd, buffer, strlen(buffer)+1, 0, (struct sockaddr*)&cli_address, len);

    recvfrom(server_fd, &cols, sizeof(cols), 0, (struct sockaddr*)&cli_address, &len);
    cols = ntohl(cols);

    // Receive each row
    for(int i = 0; i < rows; i++) {
        sprintf(buffer, "Enter row %d (space separated integers):", i+1);
        sendto(server_fd, buffer, strlen(buffer)+1, 0, (struct sockaddr*)&cli_address, len);

        recvfrom(server_fd, buffer, sizeof(buffer), 0, (struct sockaddr*)&cli_address, &len);
        buffer[strcspn(buffer, "\n")] = 0;

        char *token = strtok(buffer, " ");
        for(int j = 0; j < cols && token != NULL; j++) {
            matrix[i][j] = atoi(token);
            token = strtok(NULL, " ");
        }
    }

    // Print matrix
    printf("\nComplete Matrix:\n");
    for(int i = 0; i < rows; i++) {
        for(int j = 0; j < cols; j++)
            printf("%d ", matrix[i][j]);
        printf("\n");
    }

    close(server_fd);
    return 0;
}
