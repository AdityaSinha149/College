#include <stdio.h>
#include <stdlib.h>

int gcd(int n1,int n2){
	if(n1>n2){
		n1=n1+n2;
		n2=n1-n2;
		n1=n1-n2;
	}

	for(int i=n1;i>0;i--)
		if(n1%i==0 && n2%i==0)return i;

	return 0;
}

void main(){
	int n1,n2;
	printf("enter 2 nums");
	scanf("%d%d",&n1,&n2);
	int n3=gcd(n1,n2);
	printf("gcd of %d and %d ",n1,n2);
	n3==0? printf("DNE") :printf("is %d",n3);
}