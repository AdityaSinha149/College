#include<stdio.h>
#include<stdlib.h>
#include<math.h>

int isPrime(int p) {
    if (p <= 1) return 0;
    for (int i = 2; i <= p/2; i++)
        if (p % i == 0) return 0;
    return 1;
}

int nextPrime(int p) {
    p += 1;
    while (!isPrime(p))p++;
    return p;
}

int gcd(int n1,int n2){
    int p=2;
    int ans=1;
    while(p<n1&&p<n2){
        if(n1%p==0&&n2%p==0){
            n1/=p;
            n2/=p;
            ans*=p;
        }
        else p=nextPrime(p);
    }
    return ans;
}

void main(){
    int n1,n2;
    printf("enter 2 nums");
    scanf("%d%d",&n1,&n2);
    int n3=gcd(n1,n2);
    if(n1==0||n2==0)n3=n1+n2;
    printf("gcd of %d and %d is %d",n1,n2,n3);
}