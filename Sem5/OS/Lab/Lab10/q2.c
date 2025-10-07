#include <stdio.h>
#include <stdlib.h>
#include "mab.h"
#include "processes.h"
#include "schedules.h"

int main() {
    int n, pid, burst, priority, arrival;
    Process *queue1 = NULL, *queue2 = NULL, *queue3 = NULL;

    printf("Enter number of processes: ");
    scanf("%d", &n);

    for (int i = 0; i < n; i++) {
        printf("Process %d: Enter PID, Burst, Priority, Arrival: ", i+1);
        scanf("%d %d %d %d", &pid, &burst, &priority, &arrival);

        if (burst <= 5) push(&queue1, createProcess(pid, burst, priority, arrival));
        else if (burst <= 10) push(&queue2, createProcess(pid, burst, priority, arrival));
        else push(&queue3, createProcess(pid, burst, priority, arrival));
    }

    scheduleSJF(queue1);
    scheduleFCFS(queue2);
    schedulePriority(queue3);

    executeQueue(queue1, "SJF");
    executeQueue(queue2, "FCFS");
    executeQueue(queue3, "Priority");

    freeQueue(queue1);
    freeQueue(queue2);
    freeQueue(queue3);

    return 0;
}
