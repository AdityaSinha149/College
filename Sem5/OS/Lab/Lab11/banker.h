// banker.h — Complete Banker’s Algorithm with pthreads (header-only)

#ifndef BANKER_H
#define BANKER_H

#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>

#define P 5   // number of processes
#define R 3   // number of resources

// ----- Global variables -----
int available[R] = {3, 3, 2};
int maximum[P][R] = {
    {7, 5, 3},
    {3, 2, 2},
    {9, 0, 2},
    {2, 2, 2},
    {4, 3, 3}
};
int allocation[P][R] = {
    {0, 1, 0},
    {2, 0, 0},
    {3, 0, 2},
    {2, 1, 1},
    {0, 0, 2}
};
int need[P][R];
pthread_mutex_t lock;

// ----- Utility Functions -----

void calculate_need() {
    for (int i = 0; i < P; i++)
        for (int j = 0; j < R; j++)
            need[i][j] = maximum[i][j] - allocation[i][j];
}

int is_safe() {
    int work[R];
    int finish[P] = {0};
    int count = 0;

    for (int i = 0; i < R; i++)
        work[i] = available[i];

    while (count < P) {
        int found = 0;
        for (int i = 0; i < P; i++) {
            if (!finish[i]) {
                int j;
                for (j = 0; j < R; j++)
                    if (need[i][j] > work[j])
                        break;
                if (j == R) {
                    for (int k = 0; k < R; k++)
                        work[k] += allocation[i][k];
                    finish[i] = 1;
                    found = 1;
                    count++;
                }
            }
        }
        if (!found) return 0;
    }
    return 1;
}

int request_resources(int p, int req[]) {
    pthread_mutex_lock(&lock);

    printf("Process %d requesting: [%d %d %d]\n", p, req[0], req[1], req[2]);

    for (int i = 0; i < R; i++) {
        if (req[i] > need[p][i]) {
            printf("Error: Request > Need for P%d\n", p);
            pthread_mutex_unlock(&lock);
            return -1;
        }
        if (req[i] > available[i]) {
            printf("Process %d must wait (not enough available)\n", p);
            pthread_mutex_unlock(&lock);
            return -1;
        }
    }

    for (int i = 0; i < R; i++) {
        available[i] -= req[i];
        allocation[p][i] += req[i];
        need[p][i] -= req[i];
    }

    if (is_safe()) {
        printf("Request granted safely for P%d\n", p);
        pthread_mutex_unlock(&lock);
        return 0;
    } else {
        // rollback
        for (int i = 0; i < R; i++) {
            available[i] += req[i];
            allocation[p][i] -= req[i];
            need[p][i] += req[i];
        }
        printf("Request denied — unsafe state for P%d\n", p);
        pthread_mutex_unlock(&lock);
        return -1;
    }
}

void release_resources(int p, int rel[]) {
    pthread_mutex_lock(&lock);
    printf("Process %d releasing: [%d %d %d]\n", p, rel[0], rel[1], rel[2]);

    for (int i = 0; i < R; i++) {
        allocation[p][i] -= rel[i];
        available[i] += rel[i];
        need[p][i] += rel[i];
    }

    pthread_mutex_unlock(&lock);
}

#endif
