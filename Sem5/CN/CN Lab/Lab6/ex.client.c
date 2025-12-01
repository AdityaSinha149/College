#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORTNO 10200

int main() {
    int sd, n;
    struct sockaddr_in address;
    char buf[256];

    // Create TCP socket
    sd = socket(AF_INET, SOCK_STREAM, 0);
    if (sd < 0) {
        perror("Error creating socket");
        exit(1);
    }

    // Server details
    address.sin_family = AF_INET;
    address.sin_addr.s_addr = inet_addr("10.52.10.63"); // server IP
    address.sin_port = htons(PORTNO);

    // Connect to server
    if (connect(sd, (struct sockaddr *)&address, sizeof(address)) < 0) {
        perror("Connection failed");
        close(sd);
        exit(1);
    }

    // Input from user
    printf("Enter message: ");
    fgets(buf, sizeof(buf), stdin);

    // Send to server
    write(sd, buf, strlen(buf));

    // Receive from server
    n = read(sd, buf, sizeof(buf)-1);
    if (n > 0) {
        buf[n] = '\0'; // null terminate
        printf("The server echo is: %s\n", buf);
    }

    close(sd);
    return 0;
}
