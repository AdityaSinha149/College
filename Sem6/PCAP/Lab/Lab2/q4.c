// Title : p0->p1->p2......->pn->p0. number gets added by 1 everytime.
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

    int num;
    
    if(rank == 0){
        printf("Enter the number: ");
        fflush(stdout);
        scanf("%d", &num);
        num++;
        MPI_Send( &num , 1 , MPI_INT , 1 , 0 , MPI_COMM_WORLD);
        MPI_Recv( &num , 1 , MPI_INT , size - 1 , 0 , MPI_COMM_WORLD , &status);
        printf("Number after coming back: %d", num);
    }

    else if(rank == size - 1){
        MPI_Recv( &num , 1 , MPI_INT , rank - 1 , 0 , MPI_COMM_WORLD , &status);
        num++;
        MPI_Send( &num , 1 , MPI_INT , 0 , 0 , MPI_COMM_WORLD);
    }

    else{
        MPI_Recv( &num , 1 , MPI_INT , rank - 1 , 0 , MPI_COMM_WORLD , &status);
        num++;
        MPI_Send( &num , 1 , MPI_INT , rank + 1 , 0 , MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}