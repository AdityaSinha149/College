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
    int sockfd, newsockfd, n;
    char buf[256];
    struct sockaddr_in seraddr, cliaddr;
    socklen_t clilen;

    // Create socket
    sockfd = socket(AF_INET, SOCK_STREAM, 0);
    if (sockfd < 0) {
        perror("Error opening socket");
        exit(1);
    }

    // Server address
    seraddr.sin_family = AF_INET;
    seraddr.sin_addr.s_addr = inet_addr("10.52.10.63"); // replace with your IP
    seraddr.sin_port = htons(PORTNO);

    // Bind
    if (bind(sockfd, (struct sockaddr *)&seraddr, sizeof(seraddr)) < 0) {
        perror("Bind failed");
        exit(1);
    }

    // Listen
    listen(sockfd, 5);

    while (1) {
        clilen = sizeof(cliaddr);
        newsockfd = accept(sockfd, (struct sockaddr *)&cliaddr, &clilen);
        if (newsockfd < 0) {
            perror("Error on accept");
            continue;
        }

        if (fork() == 0) {
            // Child process
            close(sockfd); // child doesn't need the listener

            n = read(newsockfd, buf, sizeof(buf) - 1);
            if (n > 0) {
                buf[n] = '\0'; // null terminate string
                printf("\nMessage from Client: %s\n", buf);
                fflush(stdout);
                write(newsockfd, buf, strlen(buf)); // echo back
            }

            close(newsockfd);
            exit(0);
        } else {
            // Parent process
            close(newsockfd);
        }
    }

    close(sockfd);
    return 0;
}
