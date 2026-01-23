// Title  : MPI Program to Compute Factorials Using Scatter and Gather
// Author : Aditya Sinha
// Date   : 23/01/2026

#include <mpi.h>
#include <stdio.h>

long long factorial(int n) {
    long long f = 1;
    for (int i = 1; i <= n; i++) f *= i;
    return f;
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int values[size];
    int recv_val;
    long long fact;
    long long facts[size];

    if (rank == 0) {
        printf("Enter %d values:\n", size);
        for (int i = 0; i < size; i++)
            scanf("%d", &values[i]);
    }

    MPI_Scatter(values, 1, MPI_INT, &recv_val, 1, MPI_INT, 0, MPI_COMM_WORLD);

    fact = factorial(recv_val);

    MPI_Gather(&fact, 1, MPI_LONG_LONG, facts, 1, MPI_LONG_LONG, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        long long sum = 0;
        printf("Factorials:\n");
        for (int i = 0; i < size; i++) {
            printf("%lld ", facts[i]);
            sum += facts[i];
        }
        printf("\nSum of factorials = %lld\n", sum);
    }

    MPI_Finalize();
    return 0;
}
