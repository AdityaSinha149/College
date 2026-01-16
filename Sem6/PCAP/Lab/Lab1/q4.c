//Title : MPI program that calculate factorial and fibonacci of the rank if even or odd respectively
//Author : Aditya Sinha
//Date : 9/1/2026

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int fact(int n);
int fib(int n);

int main(int argc, char *argv[]){
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	if((rank & 1) == 1)
		printf("P%d prints Fib(%d) : %d\n", rank, rank, fib(rank));
	else
		printf("P%d prints Fact(%d) : %d\n", rank, rank, fact(rank));
	
	MPI_Finalize();
	return 0;
}

int fact(int n){
	return n <= 1 ? 1 : n*fact(n-1);
}

int fib(int n){
	return n <= 1 ? n : fib(n-1)+fib(n-2);
}
