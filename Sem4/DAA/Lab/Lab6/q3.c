#include <stdio.h>
#include <stdlib.h>

int inversions=0;

void merge(int *arr,int l,int m,int r){
    int n1=m-l+1;
    int n2=r-m;
    int *L=(int *)malloc(n1*sizeof(int));
    int *R=(int *)malloc(n2*sizeof(int));
    for(int i=0;i<n1;i++)L[i]=arr[l+i];
    for(int i=0;i<n2;i++)R[i]=arr[m+1+i];
    int i=0,j=0,k=l;
    while(i<n1 && j<n2){
        if(L[i]<=R[j])arr[k++]=L[i++];
        else{
            arr[k++]=R[j++];
            inversions+=n1-i;
        }
    }
    while(i<n1)arr[k++]=L[i++];
    while(j<n2)arr[k++]=R[j++];
}

void MergeSort(int *arr,int n){
    if(n<=1)return;
    int m=n/2;
    MergeSort(arr,m);
    MergeSort(arr+m,n-m);
    merge(arr,0,m-1,n-1);
}

int main(){
    int n;
    printf("Enter the number of elements: ");
    scanf("%d",&n);
    int *arr=(int *)malloc(n*sizeof(int));
    printf("Enter the elements: ");
    for(int i=0;i<n;i++)scanf("%d",arr+i);
    MergeSort(arr,n);
    printf("Sorted array: ");
    for(int i=0;i<n;i++)printf("%d ",arr[i]);
    printf("Number of inversions: %d\n",inversions);
    return 0;
}