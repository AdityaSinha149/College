// Title  : MPI Program to Compute Averages Using Scatter and Gather
// Author : Aditya Sinha
// Date   : 23/01/2026

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int M;
    float *data = NULL;
    float *recv_buf;
    float avg, avgs[size];

    if (rank == 0) {
        printf("Enter M: ");
        fflush(stdout);
        scanf("%d", &M);

        data = (float *)malloc(sizeof(float) * M * size);
        printf("Enter %d values:\n", M * size);
        for (int i = 0; i < M * size; i++)
            scanf("%f", &data[i]);
    }

    MPI_Bcast(&M, 1, MPI_INT, 0, MPI_COMM_WORLD);

    recv_buf = (float *)malloc(sizeof(float) * M);

    MPI_Scatter(data, M, MPI_FLOAT, recv_buf, M, MPI_FLOAT, 0, MPI_COMM_WORLD);

    float sum = 0;
    for (int i = 0; i < M; i++)
        sum += recv_buf[i];

    avg = sum / M;

    MPI_Gather(&avg, 1, MPI_FLOAT, avgs, 1, MPI_FLOAT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float total_avg = 0;
        printf("Averages from processes:\n");
        for (int i = 0; i < size; i++) {
            printf("%f ", avgs[i]);
            total_avg += avgs[i];
        }
        total_avg /= size;
        printf("\nTotal average = %f\n", total_avg);
        free(data);
    }

    free(recv_buf);
    MPI_Finalize();
    return 0;
}
