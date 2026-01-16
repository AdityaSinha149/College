// Title : MPI program to use synchronous send. Send a word from P0 to P1, toggle letters, and send it back.
// Author : Aditya Sinha
// Date : 16/1/2026

#include <stdio.h>
#include <mpi.h>
#include <string.h>

void toggleWord(char *s);
void toggle(char *s, int i);

int main(int argc, char *argv[]) {
    int rank;
    MPI_Status status;
    char word[100];

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("Enter the word:\n");
        scanf("%99s", word);

        MPI_Ssend(word, 100, MPI_CHAR, 1, 0, MPI_COMM_WORLD);
        MPI_Recv(word, 100, MPI_CHAR, 1, 0, MPI_COMM_WORLD, &status);

        printf("Word received from process 1: %s\n", word);
    }
    else if (rank == 1) {
        MPI_Recv(word, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD, &status);
        toggleWord(word);
        MPI_Ssend(word, 100, MPI_CHAR, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}

void toggleWord(char *s) {
    int i = 0;
    while (s[i] != '\0') {
        toggle(s, i);
        i++;
    }
}

void toggle(char *s, int i) {
    if (s[i] >= 'a' && s[i] <= 'z')
        s[i] = s[i] - 'a' + 'A';
    else if (s[i] >= 'A' && s[i] <= 'Z')
        s[i] = s[i] - 'A' + 'a';
}
