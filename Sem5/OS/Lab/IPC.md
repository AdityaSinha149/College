# **Basics**

### Header Files
```c
#include <stdio.h>      // Basic input/output
#include <stdlib.h>     // exit(), malloc(), etc.
#include <unistd.h>     // fork(), exec(), getpid(), getppid()
#include <sys/types.h>  // pid_t definition
#include <sys/wait.h>   // wait() function
```

### Important Functions
| Function    | Purpose                                        |
| ----------- | ---------------------------------------------- |
| `getpid()`  | Returns the current process ID.                |
| `getppid()` | Returns the parent process ID.                 |
| `fork()`    | Creates a new process.                         |
| `wait()`    | Makes the parent wait for its child to finish. |
| `exec()`    | Replaces the current program with another.     |
| `exit()`    | Terminates the current process.                |

### Example – `fork()` and `wait()`
```c
int main() {
    pid_t pid = fork();
    if (pid == 0)
        printf("Child Process (PID: %d)\n", getpid());
    else {
        wait(NULL);
        printf("Parent Process (PID: %d)\n", getpid());
    }
}
```
This example shows how a parent process creates a child and waits for it to finish.

---

# **Pipes**

###  Headers for Pipes
```c
#include <stdio.h>
#include <unistd.h>    // pipe(), read(), write(), fork()
#include <string.h>
#include <stdlib.h>
```

###  Key Functions
| Function                     | Description                                            |
| ---------------------------- | ------------------------------------------------------ |
| `pipe(int fd[2])`            | Creates a pipe (fd[0] for reading, fd[1] for writing). |
| `read(fd[0], buffer, size)`  | Reads data from the pipe.                              |
| `write(fd[1], buffer, size)` | Writes data into the pipe.                             |

###  Example – Simple Pipe
```c
int main() {
    int fd[2]; char msg[] = "Hello Pipe", buf[50];
    pipe(fd);

    if (fork() == 0) {
        read(fd[0], buf, sizeof(buf));
        printf("Child read: %s\n", buf);
    } else {
        write(fd[1], msg, strlen(msg) + 1);
    }
}
```
This example shows one-way communication between parent and child using a pipe.

---

# **FIFOs**
### Headers for FIFOs (Named Pipes)
```c
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
```

### Key Functions
| Function                                | Description                       |
| --------------------------------------- | --------------------------------- |
| `mkfifo(const char *path, mode_t mode)` | Creates a named pipe (FIFO).      |
| `open()` / `close()`                    | Opens and closes the FIFO.        |
| `read()` / `write()`                    | Transfers data between processes. |

### Example – FIFO Producer and Consumer
**Producer:**
```c
int main() {
    mkfifo("fifo_file", 0666);
    int fd = open("fifo_file", O_WRONLY);
    write(fd, "Hi Consumer", 12);
    close(fd);
}
```
**Consumer:**
```c
int main() {
    char buf[50];
    int fd = open("fifo_file", O_RDONLY);
    read(fd, buf, sizeof(buf));
    printf("Received: %s\n", buf);
    close(fd);
}
```
This shows two separate programs communicating through a named FIFO.

---

## ** Message Queues**

### Headers for Message Queues
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/msg.h>
```

### Key Functions
| Function                                                               | Description                          |
| ---------------------------------------------------------------------- | ------------------------------------ |
| `msgget(key_t key, int msgflg)`                                        | Creates or opens a message queue.    |
| `msgsnd(int msqid, const void *msgp, size_t msgsz, int msgflg)`        | Sends a message.                     |
| `msgrcv(int msqid, void *msgp, size_t msgsz, long msgtyp, int msgflg)` | Receives a message.                  |
| `msgctl(int msqid, int cmd, struct msqid_ds *buf)`                     | Controls or deletes a message queue. |

### Example – Message Queue
```c
struct msg { long type; char text[50]; };
int main() {
    int qid = msgget(1234, 0666 | IPC_CREAT);
    struct msg m1 = {1, "Hello MQ"};
    msgsnd(qid, &m1, sizeof(m1.text), 0);
    msgrcv(qid, &m1, sizeof(m1.text), 1, 0);
    printf("Received: %s\n", m1.text);
    msgctl(qid, IPC_RMID, NULL);
}
```
This shows simple message passing using a System V message queue.

**About Flags and Type:**
- The `msgflg` parameter (used in `msgget`, `msgsnd`, and `msgrcv`) controls message queue behavior. For example, `IPC_CREAT` creates a queue if it doesn't exist, while `IPC_EXCL` ensures an error if it already exists.
- The `type` field in the `struct msg` determines message priority and order. Lower type values can be used for general messages, and `msgrcv` can selectively read messages by specifying a particular type.

---

# **Shared Memory**
### Headers for Shared Memory
```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <unistd.h>
```

### Key Functions
| Function                                           | Description                               |
| -------------------------------------------------- | ----------------------------------------- |
| `shmget(key_t key, size_t size, int shmflg)`       | Creates or opens a shared memory segment. |
| `shmat(int shmid, const void *addr, int flag)`     | Attaches the segment to the process.      |
| `shmdt(const void *addr)`                          | Detaches the shared memory.               |
| `shmctl(int shmid, int cmd, struct shmid_ds *buf)` | Removes or controls the segment.          |

### Example – Shared Memory
```c
int main() {
    int id = shmget(1234, 1024, 0666 | IPC_CREAT);
    char *data = (char*) shmat(id, NULL, 0);
    strcpy(data, "Shared Memory Works");
    printf("Read: %s\n", data);
    shmdt(data);
    shmctl(id, IPC_RMID, NULL);
}
```
This shows how two processes can share data directly through memory.

**About Flags and Attach Mode:**
- The `shmflg` parameter in `shmget` specifies permissions and options (e.g., `IPC_CREAT` to create a segment if not existing, `0666` for read/write access).
- In `shmat`, the `flag` parameter defines attach behavior; setting it to `0` attaches normally, while using `SHM_RDONLY` makes the segment read-only.

---

## **Synchronization and Deadlock**

### Headers for POSIX Semaphores
```c
#include <stdio.h>
#include <pthread.h>
#include <semaphore.h>
#include <unistd.h>
```

### Key Functions
| Function                                                | Description                             |
| ------------------------------------------------------- | --------------------------------------- |
| `sem_init(sem_t *sem, int pshared, unsigned int value)` | Initializes a semaphore.                |
| `sem_wait(sem_t *sem)`                                  | Decreases the semaphore value (wait).   |
| `sem_post(sem_t *sem)`                                  | Increases the semaphore value (signal). |
| `sem_destroy(sem_t *sem)`                               | Destroys the semaphore.                 |

### Example – Producer Synchronization
```c
sem_t sem;
void* producer(void* arg) {
    sem_wait(&sem);
    printf("Producing...\n");
    sem_post(&sem);
}
int main() {
    pthread_t t1, t2;
    sem_init(&sem, 0, 1);
    pthread_create(&t1, NULL, producer, NULL);
    pthread_create(&t2, NULL, producer, NULL);
    pthread_join(t1, NULL);
    pthread_join(t2, NULL);
    sem_destroy(&sem);
}
```
This uses a semaphore to ensure that only one thread produces at a time.

---

### Example – Deadlock by Circular Wait
```c
pthread_mutex_t m1 = PTHREAD_MUTEX_INITIALIZER;
pthread_mutex_t m2 = PTHREAD_MUTEX_INITIALIZER;

void* t1(void* arg) {
    pthread_mutex_lock(&m1);
    sleep(1);
    pthread_mutex_lock(&m2);
}

void* t2(void* arg) {
    pthread_mutex_lock(&m2);
    sleep(1);
    pthread_mutex_lock(&m1);
}

int main() {
    pthread_t a, b;
    pthread_create(&a, NULL, t1, NULL);
    pthread_create(&b, NULL, t2, NULL);
    pthread_join(a, NULL);
    pthread_join(b, NULL);
}
```
This program intentionally creates a deadlock by acquiring locks in opposite order.

---

### Example – Sleeping Barber Problem
```c
#include <pthread.h>
#include <semaphore.h>
#include <stdio.h>
#define CHAIRS 3

sem_t customers, barber, mutex;
int waiting = 0;

void* barber_func() {
    while (1) {
        sem_wait(&customers);
        sem_wait(&mutex);
        waiting--;
        sem_post(&barber);
        sem_post(&mutex);
        printf("Barber cutting hair...\n");
        sleep(1);
    }
}

void* customer_func(void* arg) {
    sem_wait(&mutex);
    if (waiting < CHAIRS) {
        waiting++;
        printf("Customer waiting\n");
        sem_post(&customers);
        sem_post(&mutex);
        sem_wait(&barber);
    } else {
        printf("No chair available; customer leaves.\n");
        sem_post(&mutex);
    }
}

int main() {
    pthread_t b, c[5];
    sem_init(&customers, 0, 0);
    sem_init(&barber, 0, 0);
    sem_init(&mutex, 0, 1);

    pthread_create(&b, NULL, barber_func, NULL);
    for (int i = 0; i < 5; i++) {
        pthread_create(&c[i], NULL, customer_func, NULL);
        sleep(1);
    }
    pthread_join(b, NULL);
}
```
This models the sleeping barber problem where customers and a barber share limited chairs using semaphores.