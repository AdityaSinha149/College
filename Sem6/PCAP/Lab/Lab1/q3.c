//Title : MPI program that toggle Characters
//Author : Aditya Sinha
//Date : 9/1/2026

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <string.h>

char* toggle(char* s, int i);

int main(int argc, char *argv[]) {
    int rank;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char s[] = "Hello";

    printf("P%d : %s\n", rank, toggle(s, rank));

    MPI_Finalize();
    return 0;
}

char* toggle(char* s, int i) {
    if (i >= strlen(s))
        return s;

    if (s[i] >= 'a' && s[i] <= 'z')
        s[i] = s[i] - 'a' + 'A';
    else if (s[i] >= 'A' && s[i] <= 'Z')
        s[i] = s[i] - 'A' + 'a';

    return s;
}

