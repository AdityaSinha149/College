#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <stdbool.h>

#define MAX 20

struct DSA {
    int request_id;
    int arrival_time_stamp;
    int cylinder;
    int address;
    int process_id;
};

int compareArrival(const void *a, const void *b) {
    struct DSA *reqA = (struct DSA *)a;
    struct DSA *reqB = (struct DSA *)b;
    return reqA->arrival_time_stamp - reqB->arrival_time_stamp;
}

// Function to calculate total head movement for FCFS
void FCFS(struct DSA req[], int n, int head) {
    int total_movement = 0;
    printf("\n--- FCFS Disk Scheduling ---\n");
    printf("Order of service: ");
    for (int i = 0; i < n; i++) {
        printf("R%d(%d) ", req[i].request_id, req[i].cylinder);
        total_movement += abs(head - req[i].cylinder);
        head = req[i].cylinder;
    }
    printf("\nTotal Head Movement = %d\n", total_movement);
    printf("Average Head Movement = %.2f\n", (float)total_movement / n);
}

// Function to calculate total head movement for SSTF
void SSTF(struct DSA req[], int n, int head) {
    int total_movement = 0;
    bool visited[MAX] = {false};

    printf("\n--- SSTF Disk Scheduling ---\n");
    printf("Order of service: ");

    for (int count = 0; count < n; count++) {
        int min_dist = INT_MAX;
        int index = -1;
        for (int i = 0; i < n; i++) {
            if (!visited[i]) {
                int dist = abs(req[i].cylinder - head);
                if (dist < min_dist) {
                    min_dist = dist;
                    index = i;
                }
            }
        }

        visited[index] = true;
        printf("R%d(%d) ", req[index].request_id, req[index].cylinder);
        total_movement += abs(head - req[index].cylinder);
        head = req[index].cylinder;
    }

    printf("\nTotal Head Movement = %d\n", total_movement);
    printf("Average Head Movement = %.2f\n", (float)total_movement / n);
}

int main() {
    struct DSA req[MAX];
    int n, head;

    printf("Enter number of disk requests: ");
    scanf("%d", &n);

    printf("Enter initial head position: ");
    scanf("%d", &head);

    for (int i = 0; i < n; i++) {
        req[i].request_id = i + 1;
        printf("\nEnter details for Request %d:\n", req[i].request_id);
        printf("Arrival Time Stamp: ");
        scanf("%d", &req[i].arrival_time_stamp);
        printf("Cylinder Number: ");
        scanf("%d", &req[i].cylinder);
        printf("Address: ");
        scanf("%d", &req[i].address);
        printf("Process ID: ");
        scanf("%d", &req[i].process_id);
    }

    // Sort by arrival time for FCFS
    qsort(req, n, sizeof(struct DSA), compareArrival);

    FCFS(req, n, head);
    SSTF(req, n, head);

    return 0;
}
