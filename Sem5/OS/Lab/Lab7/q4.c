#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/shm.h>

typedef struct {
    int over;
    char win_msg[1024];
    char data[3][3];
    int flag;
}shm;

int win(char board[3][3], int n) {
    for(int i = 0; i < 3; i++) {
        int f = 0;
        for(int j = 0; j < 3; j++)
            if(board[i][j] != n) f = 1;
        if(f == 0) return 1;
        f = 0;
        for(int j = 0; j < 3; j++)
            if(board[j][i] != n) f = 1;
        if(f == 0) return 1;
    }
    int f = 0;
    for(int j = 0; j < 3; j++)
        if(board[j][j] != n) f = 1;
    if(f == 0) return 1;

    f = 0;
    for(int j = 0; j < 3; j++)
        if(board[j][3-j-1] != n) f = 1;
    if(f == 0) return 1;

    return 0;
}

int main() {
    int shmid = shmget(0666, sizeof(shm), 0666 | IPC_CREAT);
    if(shmid == -1) {
        perror("shmget error");
        return 1;
    }

    shm * msg = (shm *)shmat(shmid, NULL, 0);
    if(msg == (shm *)-(1)) {
        perror("shmat error");
        return 1;
    }

    printf("Enter player no : ");
    int n;
    scanf("%d", &n);

    while(1) {
        if(n == 1)
            while(msg->flag == 1);
        else
            while(msg->flag == n || msg->flag == 0);
        
        if(msg->over == 1){
            printf(msg->win_msg);
            break;
        }

        printf("Current Posn :\n");
        for(int i1 = 0; i1 < 3; i1++) {
            for(int j1 = 0; j1 < 3; j1++) {
                printf("%d ", msg->data[i1][j1]);
            }
            printf("\n");
        }
        
        printf("Enter the co-ordinate u wanna fill : ");
        int i,j;
        scanf("%d%d",&i,&j);
        
        if(i < 0 || i >= 3 || j < 0 || j >= 3){
            printf("No such cell exists");
            continue;
        }

        if(msg->data[i][j] != 0) {
            printf("Cell is already filled\n");
            continue;
        }

        msg->data[i][j] = n;
        for(int i1 = 0; i1 < 3; i1++) {
            for(int j1 = 0; j1 < 3; j1++) {
                printf("%d ", msg->data[i1][j1]);
            }
            printf("\n");
        }

        if(win(msg->data, n) == 1) {
            msg->over = 1;
            printf("You won\n");
            snprintf(msg->win_msg, sizeof(msg->win_msg), "Player %d won\n", n);
            break;
        }

        msg->flag = n;
        
    }

    msg->flag = n;

    if (n == 1) {
        if (shmctl(shmid, IPC_RMID, NULL) == -1) {
            perror("shmctl");
            return 1;
        }
    }
    
    return 0;
}