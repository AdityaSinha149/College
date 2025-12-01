#ifndef SOCKET_UTILS_H
#define SOCKET_UTILS_H

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <string.h>
#include <pthread.h>


#define PORT 10200
#define IP "172.16.48.61"

struct msg {
    char name[1024];
    char msg[1024];
};


int makeIP4Socket(){
    return socket(AF_INET, SOCK_STREAM, 0);
}

struct sockaddr_in assignSocketAddress() {
    struct sockaddr_in socketAddr;
    socketAddr.sin_port = htons(PORT);
    socketAddr.sin_family = AF_INET;
    inet_pton(AF_INET, IP, &socketAddr.sin_addr.s_addr);
    return socketAddr;
}



#endif