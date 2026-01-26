It is a message passing library interface supporting parallel programming. Data is moved from address space of one process to that of the other through cooperative operations on each process.


---------------------------------------------------------------------
--------------------------------------------------

![[mpi.1.png]]

MPI routines are implemented as fns that return exit status of the func call. MPI_SUCCESS if routine ran successfully else an integer indicsting the specific error;

## Basic
```c
MPI_Init(&argc, &argv); //initializes the mpi env
MPI_Finalize(); //terminates mpi env after releasing all the resources.
```

## Communicator
A communicator is an object that provides an environment for the processes for passing the message among each other. **MPI_COMM_WORLD** is the default communicator. We can create our own communicators if we need to partition the process into independent communication groups.

## Rank
Process in a communicator is always ordered. If there are p processes in a communicator each of the would have a rank from 0 to p-1 through which they can determine which portion of the task they have to do. It is a process's position in the overall order.

```c
MPI_Comm_rank(MPI_COMM MPI_COMM_WORLD, int *rank); //returns the rank of the process in the rank variable.
MPI_Comm_size(MPI_COMM MPI_COMM_WORLD, int *size); //returns the number of processes in a communicator in the size variable.
```

## Point-to-Point Communication in MPI
![[mpi.2.png]]

```c
MPI_Send(void *mess, int cnt, MPI_Datatype dt, int dest, int tag, MPI_Comm comm);

MPI_Recv(void *mess, int cnt, MPI_Datatype dt, int source, int tag, MPI_Comm comm, MPI_Status *status);

MPI_Ssend(void *mess, int cnt, MPI_Datatype dt, int dest, int tag, MPI_Comm comm);

MPI_Bsend(void *mess, int cnt, MPI_Datatype dt, int dest, int tag, MPI_Comm comm);
```

MPI_Send routine sends message and block the process until the application buffer in the sending task is free for reuse. The MPI implementation may buffer it allowing it to return almost immediately. If not then the send wont be complete until the matching receive occurs
### Ssend vs Bsend
Ssend blocks the process until the application buffer in the sending task is free for reuse and the dest process has started to receive the message. It is synchronous.
Bsend is buffered, it permits the programmer to allocate the req amt of buffer space into which data can be copied until its delivered. It returns when the data has been copied to the buffer.
### Fns req for BSend

```c
MPI_Buffer_attach(void *buff, int size);
MPI_Buffer_detach(void *buff, int *size);
```
**Only one buffer can be attached at a time**

## Deadlock
A process is in deadlock when it is blocked waiting for a condition that will never become true.

1. Recv-Recv
   ```c
	int a,b,c;
	int rank;
	MPI_Status status;
	if (rank==0) {
		MPI_Recv(&b,1,MPI_INT,1,0,MPI_COMM_WORLD,&status);//waiting for p1
		MPI_Send(&a,1,MPI_INT,1,0,MPI_COMM_WORLD);
		c=a+b/2;
	}
	else if (rank==1) {
		MPI_Recv(&a,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);//waiting fro p0
		MPI_Send(&b,1,MPI_INT,0,0,MPI_COMM_WORLD);
		c=a+b/2;
	}
   ```
2. Tag mismatch
   ```c
	int a,b,c;
	int rank;
	MPI_Status status;
	if(rank==0) {
		MPI_Send(&a,1,MPI_INT,1,1,MPI_COMM_WORLD);
		MPI_Recv(&b,1,MPI_INT,1,1,MPI_COMM_WORLD,&status);
		c=a+b/2;
	}
	else if(rank==1) {
		MPI_Send(&b,1,MPI_INT,0,0,MPI_COMM_WORLD);
		MPI_Recv(&a,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		c=a+b/2;
	}
   ```
3. Rank mismatch
   ```c
	int a,b,c;
	int rank;
	MPI_Status status;
	if(rank==0) {
		MPI_Send(&a,1,MPI_INT,2,1,MPI_COMM_WORLD);
		MPI_Recv(&b,1,MPI_INT,2,1,MPI_COMM_WORLD,&status);
		c=a+b/2;
	}
	else if(rank==1) {
		MPI_Send(&b,1,MPI_INT,0,0,MPI_COMM_WORLD);
		MPI_Recv(&a,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		c=a+b/2;
	}
   ```
4. Communicator Mismatch
   ```c
	int a,b,c;
	int rank;
	MPI_Status status;
	if(rank==0) {
		MPI_Send(&a,1,MPI_INT,1,1,My_Communicator);
		MPI_Recv(&b,1,MPI_INT,1,1,MPI_COMM_WORLD,&status);
		c=a+b/2;
	}
	else if(rank==1){
		MPI_Send(&b,1,MPI_INT,0,0,MPI_COMM_WORLD);
		MPI_Recv(&a,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		c=a+b/2;
	}
   ```
   5. Self blocking send
    ```c
    int a,b,c;
	int rank;
	MPI_Status status;
	if(rank==0) {
		MPI_Send(&a,1,MPI_INT,0,1,MPI_COMM_WORLD);
		MPI_Recv(&b,1,MPI_INT,1,1,MPI_COMM_WORLD,&status);
		c=a+b/2;
	}
	else if(rank==1) {
		MPI_Send(&b,1,MPI_INT,0,0,MPI_COMM_WORLD);
		MPI_Recv(&a,1,MPI_INT,0,0,MPI_COMM_WORLD,&status);
		c=a+b/2;
	}
    ```


