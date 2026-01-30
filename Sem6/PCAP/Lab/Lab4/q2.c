// Title  : MPI Program to count occurrences of a key in a matrix
// Author : Aditya Sinha
// Date   : 30/01/2026

#include <mpi.h>
#include <stdio.h>

int main(int argc, char **argv) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_set_errhandler(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int a[9], row[3], key;
    if (rank == 0) {
        for (int i = 0; i < 9; i++) scanf("%d", &a[i]);
        scanf("%d", &key);
    }

    MPI_Bcast(&key, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Scatter(a, 3, MPI_INT, row, 3, MPI_INT, 0, MPI_COMM_WORLD);

    int cnt = 0;
    for (int i = 0; i < 3; i++) if (row[i] == key) cnt++;

    int tot;
    MPI_Reduce(&cnt, &tot, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("%d\n", tot);

    MPI_Finalize();
    return 0;
} 