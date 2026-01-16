//Title : MPI program that calculate powers
//Author : Aditya Sinha
//Date : 9/1/2026

#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int mypow(int x, int y);

int main(int argc, char *argv[]){
	int rank;
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	
	int ans = mypow(5, rank);
	printf("Answer from P%d = %d\n", rank, ans);
	
	MPI_Finalize();
	return 0;
}

int mypow(int x, int y){
	if(y==0) return 1;
	if((y&1) == 1) return x*mypow(x, y-1);
	else return mypow(x*x, y/2);
}
