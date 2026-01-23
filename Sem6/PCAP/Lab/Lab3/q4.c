// Title  : MPI Program to Merge Two Strings Using Scatter and Gather
// Author : Aditya Sinha
// Date   : 23/01/2026

#include <mpi.h>
#include <stdio.h>
#include <string.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char s1[256], s2[256], result[512];
    char buf1[64], buf2[64], buf_res[128];

    int len, chunk;

    if (rank == 0) {
        printf("Enter String S1: ");
        fflush(stdout);
        scanf("%s", s1);
        printf("Enter String S2: ");
        fflush(stdout);
        scanf("%s", s2);
        len = strlen(s1);
    }

    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk = len / size;

    MPI_Scatter(s1, chunk, MPI_CHAR, buf1, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);
    MPI_Scatter(s2, chunk, MPI_CHAR, buf2, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);

    int k = 0;
    for (int i = 0; i < chunk; i++) {
        buf_res[k++] = buf1[i];
        buf_res[k++] = buf2[i];
    }

    MPI_Gather(buf_res, 2 * chunk, MPI_CHAR, result, 2 * chunk, MPI_CHAR, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        result[2 * len] = '\0';
        printf("Resultant String: %s\n", result);
    }

    MPI_Finalize();
    return 0;
}
