//Title : Calculator with MPI
//Author : Aditya Sinha
//Date : 9/1/2026

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double x = 5.3;
    double y = 4.2;

    switch (rank) {
        case 0:
            printf("P%d: x + y = %.2f\n", rank, x + y);
            break;

        case 1:
            printf("P%d: x - y = %.2f\n", rank, x - y);
            break;

        case 2:
            printf("P%d: x * y = %.2f\n", rank, x * y);
            break;

        case 3:
            printf("P%d: x / y = %.2f\n", rank, x / y);
            break;

        default:
            printf("P%d: No operation assigned\n", rank);
    }

    MPI_Finalize();
    return 0;
}

