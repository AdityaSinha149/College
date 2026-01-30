// Title  : MPI Program to form a new word by repeating each character based on its position
// Author : Aditya Sinha
// Date   : 30/01/2026

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[])
{
    int rank, size, i;
    char *word, *sub, *res;
    char ch;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        word = malloc(size + 1);
        printf("Enter word (length %d): ", size);
        scanf("%s", word);
    }

    MPI_Scatter(word, 1, MPI_CHAR,
                &ch, 1, MPI_CHAR,
                0, MPI_COMM_WORLD);

    int count = rank + 1;
    sub = malloc(count);
    for (i = 0; i < count; i++)
        sub[i] = ch;

    if (rank == 0) {
        int total = size * (size + 1) / 2;
        res = malloc(total + 1);

        int offset = 0;

        for (i = 0; i < count; i++)
            res[offset++] = sub[i];

        for (i = 1; i < size; i++) {
            MPI_Recv(res + offset, i + 1, MPI_CHAR,
                     i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            offset += i + 1;
        }

        res[offset] = '\0';
        printf("Final word: %s\n", res);
    }
    else {
        MPI_Send(sub, count, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
