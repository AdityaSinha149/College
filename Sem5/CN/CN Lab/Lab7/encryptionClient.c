#include "socketUtils.h"

int key = 3;


void encryptMessage(char * buff) {
    int m = strlen(buff);
    for (int i = 0; i < m; i++) {
        buff[i] += key;
    }
}

int main() {
    int clientFD = makeIP4Socket();
    struct sockaddr_in servAddr = assignSocketAddress();
    if(connect(clientFD, (struct sockaddr *)&servAddr, sizeof(servAddr)) == -1) {
        perror("connect error : ");
        return 1;
    }
    
    printf("Enter your msg: ");
    char msg[1024];
    fgets(msg, 1024, stdin);

    encryptMessage(msg);

    printf("encrypted msg: %s" , msg);
    
    send(clientFD, msg, 1024, 0);

    return 0;
}