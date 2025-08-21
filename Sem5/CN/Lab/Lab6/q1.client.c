#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>

#define PORT 10200
#define IP "10.154.184.195"

int main(){
    int client;
    struct sockaddr_in server_addr;
    socklen_t server_len = sizeof(server_addr);
    
    //create socket
    client = socket(AF_INET, SOCK_STREAM, 0);
    if(client < 0) {
        perror("socket error");
        exit(1);
    }

    //setup server addr
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if(inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0) {
        perror("invalid ip address");
        exit(1);
    }

    // Connect to server
    if (connect(client, (struct sockaddr *)&server_addr, server_len) < 0) {
        perror("Connection failed");
        close(client);
        exit(1);
    }

    //read inputs adn set to server
    printf("Give integers and operator in this format : a op b(separated by spaces)");
    int a,b;
    char op;
    scanf("%d %c %d", &a, &op, &b);

    int a_h = a;
    int b_h = b;

    //convert to network and send
    a = htonl(a);
    b = htonl(b);

    send(client, &a, sizeof(a), 0);
    send(client, &op, sizeof(op), 0);
    send(client, &b, sizeof(b), 0);

    //receive ans and convert to host
    int ans;
    recv(client, &ans, sizeof(ans), 0);
    ans = ntohl(ans);

    printf("%d %c %d = %d\n", a_h, op, b_h, ans);
    close(client);
    return 0;
}