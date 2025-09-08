#include "socketUtils.h"

int key = 3;

void decryptMessage(char * buff) {
    int m = strlen(buff);
    for (int i = 0; i < m; i++) {
        buff[i] -= key;
    }
}

int main() {
    int serverFD = makeIP4Socket();
    struct sockaddr_in servAddr = assignSocketAddress();
    
    if(bind(serverFD, (struct sockaddr *)&servAddr, sizeof(servAddr)) == -1) {
        perror("bind error");
        return 1;
    }

    if(listen(serverFD, 10) == -1) {
        perror("listen error");
        return 1;
    }

    printf("Server listening at %s:%d\n", IP, PORT);

    // Accept a client connection
    int clientFD = accept(serverFD, NULL, NULL);
    if (clientFD == -1) {
        perror("accept error");
        return 1;
    }

    char msg[1024];
    int n = recv(clientFD, msg, sizeof(msg) - 1, 0);
    if (n > 0) {
        msg[n] = '\0'; // Null terminate
        decryptMessage(msg);
        printf("Decrypted msg: %s\n", msg);
    } else {
        printf("No data received.\n");
    }

    close(clientFD);
    close(serverFD);

    return 0;
}
