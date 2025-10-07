#ifndef PROCESSES_H
#define PROCESSES_H

#include <stdio.h>
#include <stdlib.h>

typedef struct process {
    int pid;
    int burst;
    int priority;
    int arrival;
    struct process *next;
} Process;

static Process* createProcess(int pid, int burst, int priority, int arrival) {
    Process* p = (Process*)malloc(sizeof(Process));
    p->pid = pid;
    p->burst = burst;
    p->priority = priority;
    p->arrival = arrival;
    p->next = NULL;
    return p;
}

static void push(Process** head, Process* p) {
    if (!*head) *head = p;
    else {
        Process* temp = *head;
        while (temp->next) temp = temp->next;
        temp->next = p;
    }
}

static void freeQueue(Process* head) {
    while (head) {
        Process* temp = head;
        head = head->next;
        free(temp);
    }
}

#endif
