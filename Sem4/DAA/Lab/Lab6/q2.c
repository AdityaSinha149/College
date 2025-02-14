#include <stdio.h>
#include <stdlib.h>

void swap(int *a,int *b){
    *a^=*b;
    *b^=*a;
    *a^=*b;
}

void QuickSort(int *arr,int n){
    int p=arr[0];
    int i=1;
    int j=n-1;
    while(i<j){
        while(arr[i]<=p && i<n)i++;
        while(arr[j]>p && j>0)j--;
        if(i<j)swap(&arr[i],&arr[j]);
    }
    if(j>0)swap(&arr[0],&arr[j]);
    if(j>1)QuickSort(arr,j);
    //why j>1? because if j=1, then there is only one element in the array, so no need to sort
    if(j<n-1)QuickSort(arr+j+1,n-j-1);
    //why j<n-1? because if j=n-1, then there is only one element in the array, so no need to sort
}

int main(){
    int n;
    printf("Enter the number of elements: ");
    scanf("%d",&n);
    int *arr=(int *)malloc(n*sizeof(int));
    printf("Enter the elements: ");
    for(int i=0;i<n;i++)scanf("%d",arr+i);
    QuickSort(arr,n);
    printf("Sorted array: ");
    for(int i=0;i<n;i++)printf("%d ",arr[i]);
    printf("\n");
    return 0;
}