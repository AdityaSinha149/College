// Title  : MPI Program to Count Non-Vowels Using Scatter and Gather
// Author : Aditya Sinha
// Date   : 23/01/2026

#include <mpi.h>
#include <stdio.h>
#include <string.h>
#include <ctype.h>

int is_vowel(char c) {
    c = tolower(c);
    return (c=='a'||c=='e'||c=='i'||c=='o'||c=='u');
}

int main(int argc, char *argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    char str[256];
    char recv_buf[64];
    int counts[size];

    int len, chunk;
    int count = 0;

    if (rank == 0) {
        printf("Enter string: ");
        fflush(stdout);
        scanf("%s", str);
        len = strlen(str);
    }

    MPI_Bcast(&len, 1, MPI_INT, 0, MPI_COMM_WORLD);
    chunk = len / size;

    MPI_Scatter(str, chunk, MPI_CHAR, recv_buf, chunk, MPI_CHAR, 0, MPI_COMM_WORLD);

    for (int i = 0; i < chunk; i++) {
        if (!is_vowel(recv_buf[i]))
            count++;
    }

    MPI_Gather(&count, 1, MPI_INT, counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        int total = 0;
        printf("Non-vowels by each process:\n");
        for (int i = 0; i < size; i++) {
            printf("%d ", counts[i]);
            total += counts[i];
        }
        printf("\nTotal non-vowels = %d\n", total);
    }

    MPI_Finalize();
    return 0;
}
