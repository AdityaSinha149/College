#ifndef SCHEDULES_H
#define SCHEDULES_H

#include "processes.h"

static void scheduleSJF(Process* head) {
    for (Process* i = head; i; i = i->next) {
        for (Process* j = i->next; j; j = j->next) {
            if (i->burst > j->burst) {
                int tmp = i->pid; i->pid = j->pid; j->pid = tmp;
                tmp = i->burst; i->burst = j->burst; j->burst = tmp;
                tmp = i->priority; i->priority = j->priority; j->priority = tmp;
                tmp = i->arrival; i->arrival = j->arrival; j->arrival = tmp;
            }
        }
    }
}

static void scheduleFCFS(Process* head) {
    for (Process* i = head; i; i = i->next) {
        for (Process* j = i->next; j; j = j->next) {
            if (i->arrival > j->arrival) {
                int tmp = i->pid; i->pid = j->pid; j->pid = tmp;
                tmp = i->burst; i->burst = j->burst; j->burst = tmp;
                tmp = i->priority; i->priority = j->priority; j->priority = tmp;
                tmp = i->arrival; i->arrival = j->arrival; j->arrival = tmp;
            }
        }
    }
}

static void schedulePriority(Process* head) {
    for (Process* i = head; i; i = i->next) {
        for (Process* j = i->next; j; j = j->next) {
            if (i->priority < j->priority) {
                int tmp = i->pid; i->pid = j->pid; j->pid = tmp;
                tmp = i->burst; i->burst = j->burst; j->burst = tmp;
                tmp = i->priority; i->priority = j->priority; j->priority = tmp;
                tmp = i->arrival; i->arrival = j->arrival; j->arrival = tmp;
            }
        }
    }
}

static void executeQueue(Process* head, const char* name) {
    printf("\n--- Executing %s Queue ---\n", name);
    Process* curr = head;
    while (curr) {
        printf("PID: %d | Burst: %d | Priority: %d | Arrival: %d\n",
               curr->pid, curr->burst, curr->priority, curr->arrival);
        curr = curr->next;
    }
}

#endif
