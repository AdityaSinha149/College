// Title  : MPI Program to compute the sum of fact of ranks
// Author : Aditya Sinha
// Date   : 30/01/2026

#include<stdio.h>
#include<mpi.h>

int factorial(int num){
	if(num==0 || num==1){
		return 1;

	}
	return num*factorial(num-1);
}

void ErrorHandler(int error_code);

int main(int argc, char*argv[]){

	int rank,size, error_code;

	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	int fact=factorial(rank+1);

	int sum;

	MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);

	error_code=MPI_Scan(&fact, &sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
	ErrorHandler(error_code);

	if(rank == size - 1){
		fprintf(stdout,"1! + 2! +...+ %d!=%d \n", size, sum);
		fflush(stdout);
	}
	
	MPI_Finalize();
	return 0;
}

void ErrorHandler( int error_code){

	if(error_code != MPI_SUCCESS){
		char error_string[BUFSIZ];

		int length_of_error_string, error_class;

		MPI_Error_class(error_code, &error_class);
		MPI_Error_string(error_code, error_string, &length_of_error_string);

		fprintf(stderr, "%d %s\n", error_class, error_string);
	}
}