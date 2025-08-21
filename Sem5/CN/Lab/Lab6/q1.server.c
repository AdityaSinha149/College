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

int eval(int a, char op, int b){
    switch(op){
        case '+' : return a + b;
        case '-' : return a - b;
        case '%' : return a % b;
        case '*' : return a * b;
        case '/' : return a / b;
    }
}

int main(){
    int server,client;
    struct sockaddr_in server_addr, client_addr;
    socklen_t server_len = sizeof(server_addr);
    socklen_t client_len = sizeof(client_addr);

    //make socket
    server = socket(AF_INET, SOCK_STREAM, 0);
    if(server < 0) {
        perror("server error");
        exit(1);
    }

    //set server addr
    server_addr.sin_family = AF_INET;
    server_addr.sin_port = htons(PORT);
    if(inet_pton(AF_INET, IP, &server_addr.sin_addr) <= 0) {
        perror("invalid ip address");
        close(server);
        exit(1);
    }

    //bind the server with socket
    if(bind(server, (struct sockaddr*)&server_addr, server_len)) {
        perror("bind error");
        close(server);
        exit(1);
    }
    
    printf("Server listening at %s:%d\n", IP, PORT);

    //listen to clients
    if(listen(server, 5) < 0) {
        perror("listen error");
        close(server);
        exit(1);
    }
    while(1) {
        //accept any connections
        client = accept(server, (struct sockaddr*)&client_addr, &client_len);
        if(client < 0) {
            perror("accept error");
            close(server);
            exit(1);
        }
        
        //child process
        if(fork() == 0){
            //no need to listen to parents
            close(server);

            int a,b;
            char op;
            recv(client, &a, sizeof(a), 0);
            recv(client, &op, sizeof(op), 0);
            recv(client, &b, sizeof(b), 0);

            //convert to host again
            a = ntohl(a);
            b = ntohl(b);

            printf("Client asked %d %c %d = \?\n", a, op, b);

            int ans = eval(a, op, b);
            //convert ans to network and send
            ans = htonl(ans);
            send(client, &ans, sizeof(ans), 0);

            //ok byebye client
            close(client);
            exit(1);
        }
        else{
            close(client);
        }
    }

    close(server);
    return 0;
    
}