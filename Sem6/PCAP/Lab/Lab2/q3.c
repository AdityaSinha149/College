// Title : Master process sends a number and even and odd ranked slaves print square and cube of the number respectively.
// Author : Aditya Sinha
// Date : 16/1/2026

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Status status;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *num = malloc(sizeof(int)*(size));

    int buffer_size = size * (sizeof(int) + MPI_BSEND_OVERHEAD);
    void *buffer = malloc(buffer_size);
    MPI_Buffer_attach(buffer, buffer_size);

    if (rank == 0) {
        for(int i = 1; i < size; i++) {
            printf("Enter the number for process%d: ", i);
            fflush(stdout);
            scanf("%d", num+i);
        }

        for (int i = 1; i < size; i++) {
            MPI_Bsend(num+i, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
        }
    } else {
        MPI_Recv(num + rank, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, &status);
        printf("Process %d processed answer: %d\n", rank, (rank&1) == 0 ? num[rank]*num[rank] : num[rank]*num[rank]*num[rank]);
    }

    MPI_Buffer_detach(&buffer, &buffer_size);
    free(buffer);

    MPI_Finalize();
    return 0;
}
