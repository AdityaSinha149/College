// Title  : MPI Program to compute the sum of cols till the current row
// Author : Aditya Sinha
// Date   : 30/01/2026

#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int a[16], row[4], out[16], res[4];
    if (rank == 0) for (int i = 0; i < 16; i++) scanf("%d", &a[i]);

    MPI_Scatter(a, 4, MPI_INT, row, 4, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scan(row, res, 4, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    MPI_Gather(res, 4, MPI_INT, out, 4, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) printf("%d ", out[i*4 + j]);
            printf("\n");
        }
    }

    MPI_Finalize();
    return 0;
} 